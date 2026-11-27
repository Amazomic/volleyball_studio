use std::cell::RefCell;

const MAX_RALLIES: usize = 1_000_000;
const MAX_SAFE_FRAME: f64 = 9_007_199_254_740_990.0;

#[derive(Default)]
struct RallyIndex {
    bounds: Vec<[f64; 2]>,
    committed: bool,
}

thread_local! {
    static INDEX: RefCell<RallyIndex> = RefCell::new(RallyIndex::default());
}

#[no_mangle]
pub extern "C" fn volley_preview_rally_reserve(count: u32) -> *mut f64 {
    INDEX.with(|cell| {
        let mut index = cell.borrow_mut();
        index.committed = false;
        index.bounds.clear();
        let count = count as usize;
        if count > MAX_RALLIES || index.bounds.try_reserve_exact(count).is_err() {
            return std::ptr::null_mut();
        }
        index.bounds.resize(count, [f64::NAN; 2]);
        index.bounds.as_mut_ptr().cast()
    })
}

#[no_mangle]
pub extern "C" fn volley_preview_rally_commit() -> i32 {
    INDEX.with(|cell| {
        let mut index = cell.borrow_mut();
        let valid_frame =
            |frame: f64| (0.0..=MAX_SAFE_FRAME).contains(&frame) && frame.fract() == 0.0;
        index.committed = index
            .bounds
            .iter()
            .all(|&[start, end]| valid_frame(start) && valid_frame(end) && start <= end)
            && index.bounds.windows(2).all(|pair| pair[0][0] <= pair[1][0]);
        i32::from(index.committed)
    })
}

#[no_mangle]
pub extern "C" fn volley_preview_rally_at(frame: f64) -> i32 {
    INDEX.with(|cell| {
        let index = cell.borrow();
        if !index.committed || !frame.is_finite() {
            return -1;
        }
        let left = index.bounds.partition_point(|bounds| bounds[0] <= frame);
        left.checked_sub(1)
            .filter(|&i| frame < index.bounds[i][1] + 1.0)
            .map_or(-1, |i| i as i32)
    })
}

#[no_mangle]
pub extern "C" fn volley_preview_rally_neighbor(frame: f64, direction: i32) -> i32 {
    INDEX.with(|cell| {
        let index = cell.borrow();
        if !index.committed || !frame.is_finite() {
            return -1;
        }
        let left = index.bounds.partition_point(|bounds| bounds[0] <= frame);
        let candidate = match direction {
            -1 => left.checked_sub(1),
            1 => Some(left),
            _ => None,
        };
        candidate
            .filter(|&i| i < index.bounds.len())
            .map_or(-1, |i| i as i32)
    })
}

pub(crate) fn next_rally(input: &serde_json::Value) -> serde_json::Value {
    use serde_json::{json, Value};
    let mut rows: Vec<_> = input["rows"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(i, row)| {
            row["start"]
                .as_f64()
                .filter(|v| v.is_finite())
                .map(|start| (i, start, &row["id"]))
        })
        .collect();
    rows.sort_by(|a, b| a.1.total_cmp(&b.1));
    if input["first"].as_bool() == Some(true) {
        return json!(rows.first().map(|row| row.0));
    }
    if let Some(position) = rows.iter().position(|row| *row.2 == input["current"]) {
        return json!(rows.get(position + 1).map(|row| row.0));
    }
    let Some(fps) = input["fps"].as_f64().filter(|v| v.is_finite() && *v > 0.0) else {
        return Value::Null;
    };
    let Some(boundary) = input["boundary"].as_f64().filter(|v| v.is_finite()) else {
        return Value::Null;
    };
    let offset = input["offset"].as_f64().unwrap_or_default();
    json!(rows
        .iter()
        .find(|row| offset + row.1 / fps >= boundary - 0.02)
        .map(|row| row.0))
}

#[derive(Default)]
struct TouchIndex {
    frames: Vec<f64>,
    committed: bool,
    end: i32,
}

thread_local! {
    static TOUCH_INDEX: RefCell<TouchIndex> = RefCell::new(TouchIndex::default());
}

#[no_mangle]
pub extern "C" fn volley_preview_touch_reserve(count: u32) -> *mut f64 {
    TOUCH_INDEX.with(|cell| {
        let mut index = cell.borrow_mut();
        index.committed = false;
        index.end = -1;
        index.frames.clear();
        let count = count as usize;
        if count > MAX_RALLIES || index.frames.try_reserve_exact(count).is_err() {
            return std::ptr::null_mut();
        }
        index.frames.resize(count, f64::NAN);
        index.frames.as_mut_ptr()
    })
}

#[no_mangle]
pub extern "C" fn volley_preview_touch_commit() -> i32 {
    TOUCH_INDEX.with(|cell| {
        let mut index = cell.borrow_mut();
        index.committed = index
            .frames
            .iter()
            .all(|&frame| (0.0..=MAX_SAFE_FRAME).contains(&frame) && frame.fract() == 0.0)
            && index.frames.windows(2).all(|pair| pair[0] <= pair[1]);
        index.end = -1;
        i32::from(index.committed)
    })
}

#[no_mangle]
pub extern "C" fn volley_preview_touch_range(frame: f64, radius: f64) -> i32 {
    TOUCH_INDEX.with(|cell| {
        let mut index = cell.borrow_mut();
        index.end = -1;
        if !index.committed || !frame.is_finite() || !radius.is_finite() || radius < 0.0 {
            return -1;
        }
        let start = index
            .frames
            .partition_point(|&value| value < frame - radius);
        index.end = index
            .frames
            .partition_point(|&value| value <= frame + radius) as i32;
        start as i32
    })
}

#[no_mangle]
pub extern "C" fn volley_preview_touch_end() -> i32 {
    TOUCH_INDEX.with(|cell| cell.borrow().end)
}
