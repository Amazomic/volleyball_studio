use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{Scope, ScopedJoinHandle};

type Completed<R> = (u64, R);

pub(super) struct FrameQueue<T, R> {
    free: Vec<T>,
    sender: SyncSender<(u64, T)>,
    receiver: Receiver<(u64, T, R)>,
    pending: usize,
}

impl<T: Send, R: Send> FrameQueue<T, R> {
    pub(super) fn start<'scope, 'env: 'scope>(
        scope: &'scope Scope<'scope, 'env>,
        buffers: [T; 2],
        mut infer: impl FnMut(u64, &T) -> Result<R, String> + Send + 'scope,
    ) -> (Self, ScopedJoinHandle<'scope, Result<(), String>>)
    where
        T: 'scope,
        R: 'scope,
    {
        let (sender, jobs) = mpsc::sync_channel::<(u64, T)>(1);
        let (finished, receiver) = mpsc::channel();
        let worker = scope.spawn(move || {
            while let Ok((frame, input)) = jobs.recv() {
                let output = infer(frame, &input)?;
                if finished.send((frame, input, output)).is_err() {
                    break;
                }
            }
            Ok(())
        });
        (
            Self {
                free: Vec::from(buffers),
                sender,
                receiver,
                pending: 0,
            },
            worker,
        )
    }

    fn take_result(&mut self) -> Result<(u64, T, R), String> {
        if self.pending == 0 {
            return Err("RF-DETR queue has no pending frame".into());
        }
        let result = self.receiver.recv().map_err(|_| "RF-DETR worker stopped")?;
        self.pending -= 1;
        Ok(result)
    }

    pub(super) fn acquire(&mut self) -> Result<(T, Option<Completed<R>>), String> {
        if let Some(input) = self.free.pop() {
            return Ok((input, None));
        }
        let (frame, input, output) = self.take_result()?;
        Ok((input, Some((frame, output))))
    }

    pub(super) fn release(&mut self, input: T) {
        self.free.push(input);
    }

    pub(super) fn submit(&mut self, frame: u64, input: T) -> Result<(), String> {
        if self.pending >= 2 {
            return Err("RF-DETR queue capacity exceeded".into());
        }
        self.sender
            .send((frame, input))
            .map_err(|_| "RF-DETR worker stopped")?;
        self.pending += 1;
        Ok(())
    }

    pub(super) fn receive(&mut self) -> Result<Completed<R>, String> {
        let (frame, input, output) = self.take_result()?;
        self.release(input);
        Ok((frame, output))
    }

    pub(super) fn pending(&self) -> usize {
        self.pending
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn second_input_can_be_prepared_before_first_inference_finishes() {
        std::thread::scope(|scope| {
            let (started_tx, started_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let (mut queue, worker) = FrameQueue::start(scope, [10u64, 20], move |frame, input| {
                if frame == 3 {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                }
                Ok(frame + *input)
            });
            let (first, done) = queue.acquire().unwrap();
            assert!(done.is_none());
            queue.submit(3, first).unwrap();
            started_rx.recv().unwrap();
            let (second, done) = queue.acquire().unwrap();
            assert!(done.is_none());
            queue.submit(4, second).unwrap();
            release_tx.send(()).unwrap();
            let (returned, done) = queue.acquire().unwrap();
            assert_eq!(done, Some((3, first + 3)));
            assert_eq!(returned, first);
            queue.release(returned);
            assert_eq!(queue.receive().unwrap(), (4, second + 4));
            assert_eq!(queue.pending(), 0);
            drop(queue);
            worker.join().unwrap().unwrap();
        });
    }

    #[test]
    fn failing_inference_never_returns_a_successful_frame() {
        std::thread::scope(|scope| {
            let (mut queue, worker) =
                FrameQueue::<u8, u8>::start(scope, [1, 2], |_, _| Err("inference failed".into()));
            let (input, _) = queue.acquire().unwrap();
            queue.submit(0, input).unwrap();
            assert!(queue.receive().is_err());
            drop(queue);
            assert_eq!(worker.join().unwrap().unwrap_err(), "inference failed");
        });
    }
}
