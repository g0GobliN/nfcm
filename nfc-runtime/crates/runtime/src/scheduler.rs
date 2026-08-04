//! Lightweight job scheduler for compile / load / unload work.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

pub type JobId = Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerState {
    Idle,
    Queued,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerJob {
    Compile { prompt: String },
    Load { model_id: Uuid },
    Unload,
    OptimizeMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledJob {
    pub id: JobId,
    pub job: SchedulerJob,
    pub state: SchedulerState,
    pub message: String,
}

#[derive(Default)]
pub struct Scheduler {
    queue: VecDeque<ScheduledJob>,
    history: Vec<ScheduledJob>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&mut self, job: SchedulerJob) -> JobId {
        let id = Uuid::new_v4();
        self.queue.push_back(ScheduledJob {
            id,
            job,
            state: SchedulerState::Queued,
            message: String::new(),
        });
        id
    }

    pub fn pop(&mut self) -> Option<ScheduledJob> {
        let mut job = self.queue.pop_front()?;
        job.state = SchedulerState::Running;
        Some(job)
    }

    pub fn complete(&mut self, mut job: ScheduledJob, message: impl Into<String>) {
        job.state = SchedulerState::Completed;
        job.message = message.into();
        self.history.push(job);
    }

    pub fn fail(&mut self, mut job: ScheduledJob, message: impl Into<String>) {
        job.state = SchedulerState::Failed;
        job.message = message.into();
        self.history.push(job);
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    pub fn history(&self) -> &[ScheduledJob] {
        &self.history
    }

    pub fn is_idle(&self) -> bool {
        self.queue.is_empty()
    }
}
