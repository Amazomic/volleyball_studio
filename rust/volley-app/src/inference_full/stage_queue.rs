use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{oneshot, Semaphore, SemaphorePermit};

const CANCEL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Default)]
pub(crate) struct VideoTurn {
    detector_before: Option<oneshot::Receiver<()>>,
    detector_after: Option<oneshot::Sender<()>>,
    publish_before: Option<oneshot::Receiver<()>>,
    publish_after: Option<oneshot::Sender<()>>,
}

pub(crate) fn video_turns(count: usize) -> Vec<VideoTurn> {
    let mut turns: Vec<VideoTurn> = (0..count).map(|_| VideoTurn::default()).collect();
    for i in 1..count {
        let (tx, rx) = oneshot::channel();
        turns[i - 1].detector_after = Some(tx);
        turns[i].detector_before = Some(rx);
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
    pub(crate) async fn wait_detector(&mut self, cancel: &AtomicBool) -> Result<(), String> {
        wait_turn(&mut self.detector_before, cancel).await
    }
    pub(crate) fn finish_detector(&mut self) {
        if let Some(tx) = self.detector_after.take() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn next_ball_waits_for_detector_but_not_for_publication() {
        let cancel = AtomicBool::new(false);
        let mut turns = video_turns(2).into_iter();
        let mut first = turns.next().unwrap();
        let mut second = turns.next().unwrap();
        first.wait_detector(&cancel).await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), second.wait_detector(&cancel))
                .await
                .is_err()
        );
        first.finish_detector();
        second.wait_detector(&cancel).await.unwrap();
        second.finish_detector();
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
        assert!(second.wait_detector(&cancel).await.is_err());
        cancel.store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(third.wait_detector(&cancel).await.unwrap_err(), "cancelled");
    }
}
