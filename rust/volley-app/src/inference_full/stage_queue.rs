use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{oneshot, Semaphore, SemaphorePermit};

const CANCEL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Default)]
pub(crate) struct VideoTurn {
    admit_before: Option<oneshot::Receiver<()>>,
    admit_after: Option<oneshot::Sender<()>>,
    ball_before: Option<oneshot::Receiver<()>>,
    ball_after: Option<oneshot::Sender<()>>,
    publish_before: Option<oneshot::Receiver<()>>,
    publish_after: Option<oneshot::Sender<()>>,
}

pub(crate) fn video_turns(count: usize) -> Vec<VideoTurn> {
    let mut turns: Vec<VideoTurn> = (0..count).map(|_| VideoTurn::default()).collect();
    for i in 1..count {
        let (tx, rx) = oneshot::channel();
        turns[i - 1].admit_after = Some(tx);
        turns[i].admit_before = Some(rx);
        let (tx, rx) = oneshot::channel();
        turns[i - 1].ball_after = Some(tx);
        turns[i].ball_before = Some(rx);
        let (tx, rx) = oneshot::channel();
        turns[i - 1].publish_after = Some(tx);
        turns[i].publish_before = Some(rx);
    }
    turns
}

async fn wait_turn(
    before: &mut Option<oneshot::Receiver<()>>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    if let Some(rx) = before.as_mut() {
        loop {
            if cancel.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
            match tokio::time::timeout(CANCEL_INTERVAL, &mut *rx).await {
                Ok(Ok(())) => break,
                Ok(Err(_)) => return Err("Предыдущая стадия очереди не завершена".into()),
                Err(_) => {}
            }
        }
        *before = None;
    }
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    Ok(())
}

impl VideoTurn {
    pub(crate) async fn wait_ball_admission(&mut self, cancel: &AtomicBool) -> Result<(), String> {
        wait_turn(&mut self.admit_before, cancel).await
    }

    pub(crate) fn admit_ball(&mut self) {
        if let Some(tx) = self.admit_after.take() {
            let _ = tx.send(());
        }
    }

    pub(crate) async fn wait_ball(&mut self, cancel: &AtomicBool) -> Result<(), String> {
        wait_turn(&mut self.ball_before, cancel).await
    }
    pub(crate) fn finish_ball(&mut self) {
        self.admit_ball();
        if let Some(tx) = self.ball_after.take() {
            let _ = tx.send(());
        }
    }
    pub(crate) async fn wait_publish(&mut self, cancel: &AtomicBool) -> Result<(), String> {
        wait_turn(&mut self.publish_before, cancel).await
    }
    pub(crate) fn finish_publish(&mut self) {
        if let Some(tx) = self.publish_after.take() {
            let _ = tx.send(());
        }
    }
}

pub(crate) async fn acquire_gpu<'a>(
    gpu: &'a Semaphore,
    cancel: &AtomicBool,
) -> Result<SemaphorePermit<'a>, String> {
    let request = gpu.acquire();
    tokio::pin!(request);
    loop {
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        if let Ok(permit) = tokio::time::timeout(CANCEL_INTERVAL, &mut request).await {
            let permit = permit.map_err(|e| e.to_string())?;
            if cancel.load(Ordering::Acquire) {
                return Err("cancelled".into());
            }
            return Ok(permit);
        }
    }
}

pub(crate) struct GpuStages<'a> {
    gpu: &'a Semaphore,
    permit: Option<SemaphorePermit<'a>>,
    turn: &'a mut VideoTurn,
}

impl<'a> GpuStages<'a> {
    pub(crate) fn new(
        gpu: &'a Semaphore,
        permit: SemaphorePermit<'a>,
        turn: &'a mut VideoTurn,
    ) -> Self {
        Self {
            gpu,
            permit: Some(permit),
            turn,
        }
    }

    pub(crate) fn finish_ball(&mut self) {
        self.turn.finish_ball();
        self.permit.take();
    }

    pub(crate) fn enter_people(&mut self, cancel: &AtomicBool) -> Result<(), String> {
        self.permit =
            Some(tokio::runtime::Handle::current().block_on(acquire_gpu(self.gpu, cancel))?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn next_ball_is_released_before_previous_video_publication() {
        let cancel = AtomicBool::new(false);
        let mut turns = video_turns(2).into_iter();
        let mut first = turns.next().unwrap();
        let mut second = turns.next().unwrap();
        first.wait_ball(&cancel).await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), second.wait_ball(&cancel))
                .await
                .is_err()
        );
        first.finish_ball();
        second.wait_ball(&cancel).await.unwrap();
        second.finish_ball();
        first.wait_publish(&cancel).await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), second.wait_publish(&cancel))
                .await
                .is_err()
        );
        first.finish_publish();
        second.wait_publish(&cancel).await.unwrap();
        second.finish_publish();
    }

    #[tokio::test]
    async fn dropped_predecessor_fails_closed_and_cancel_unblocks_waiters() {
        let cancel = AtomicBool::new(false);
        let mut turns = video_turns(3).into_iter();
        let first = turns.next().unwrap();
        let mut second = turns.next().unwrap();
        let mut third = turns.next().unwrap();
        drop(first);
        assert!(second.wait_ball(&cancel).await.is_err());
        cancel.store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(third.wait_ball(&cancel).await.unwrap_err(), "cancelled");
    }

    #[tokio::test]
    async fn ready_next_ball_claims_gpu_before_previous_players() {
        let cancel = AtomicBool::new(false);
        let gpu = Semaphore::new(1);
        let mut turns = video_turns(2).into_iter();
        let mut first = turns.next().unwrap();
        let mut second = turns.next().unwrap();
        first.wait_ball_admission(&cancel).await.unwrap();
        let first_permit = acquire_gpu(&gpu, &cancel).await.unwrap();
        first.wait_ball(&cancel).await.unwrap();
        first.admit_ball();
        second.wait_ball_admission(&cancel).await.unwrap();
        let next_ball = acquire_gpu(&gpu, &cancel);
        tokio::pin!(next_ball);
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut next_ball)
                .await
                .is_err()
        );
        first.finish_ball();
        drop(first_permit);
        let next_permit = next_ball.await.unwrap();
        second.wait_ball(&cancel).await.unwrap();
        let players = acquire_gpu(&gpu, &cancel);
        tokio::pin!(players);
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut players)
                .await
                .is_err()
        );
        second.finish_ball();
        drop(next_permit);
        let permit = players.await.unwrap();
        assert_eq!(gpu.available_permits(), 0);
        drop(permit);
        assert_eq!(gpu.available_permits(), 1);
    }

    #[tokio::test]
    async fn gpu_permit_is_exclusive_and_released_at_ball_boundary() {
        let cancel = AtomicBool::new(false);
        let gpu = tokio::sync::Semaphore::new(1);
        let mut turns = video_turns(2).into_iter();
        let mut first = turns.next().unwrap();
        let mut second = turns.next().unwrap();
        first.wait_ball(&cancel).await.unwrap();
        let permit = acquire_gpu(&gpu, &cancel).await.unwrap();
        let mut stages = GpuStages::new(&gpu, permit, &mut first);
        assert_eq!(gpu.available_permits(), 0);
        stages.finish_ball();
        second.wait_ball(&cancel).await.unwrap();
        let next = acquire_gpu(&gpu, &cancel).await.unwrap();
        assert_eq!(gpu.available_permits(), 0);
        drop(next);
        drop(stages);
        assert_eq!(gpu.available_permits(), 1);
    }
}
