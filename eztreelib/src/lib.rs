#![feature(core_intrinsics)]
use eytzinger::SliceExt;
use ginterval::Interval;

#[derive(Debug)]
pub struct Node<T: Default + std::fmt::Debug> {
    interval: Interval<T>,
    max: u32,
}

#[derive(Debug)]
pub struct EzTree<T: Default + std::fmt::Debug> {
    // FIXME: make not public and fix test that relies on this
    pub intervals: Vec<Node<T>>,
    #[cfg(feature = "nightly")]
    mask: usize,
    #[cfg(feature = "nightly")]
    offset: usize,
    #[cfg(feature = "nightly")]
    multiplier: usize,
}

#[derive(Debug)]
struct StackCell<'a, T: Default + std::fmt::Debug> {
    node: &'a Node<T>,
    left_checked: bool,
    index: usize,
    //tree_level: usize,
}

// Heavily based off of https://github.com/jonhoo/ordsearch/blob/master/src/lib.rs
impl<T: Default + std::fmt::Debug> EzTree<T> {
    pub fn new(mut intervals: Vec<Interval<T>>) -> Self {
        intervals.sort();
        // TODO: Maybe pre walk it or get the index's permuations and do it backward, add max ends
        // that way
        intervals.eytzingerize(&mut eytzinger::permutation::InplacePermutator);
        let mut intervals = intervals
            .into_iter()
            .map(|iv| Node {
                interval: iv,
                max: 0,
            })
            .collect();
        Self::set_max(&mut intervals, 0);

        // Maybe nightly only
        #[cfg(feature = "nightly")]
        {
            use std::mem;
            let multiplier = 64 / mem::size_of::<Interval<T>>();
            let offset = multiplier + multiplier / 2;
            let mut mask = 1;
            while mask <= intervals.len() {
                mask <<= 1;
            }
            mask -= 1;
            EzTree {
                intervals,
                mask: mask,
                offset: offset,
                multiplier: multiplier,
            }
        }
        #[cfg(not(feature = "nightly"))]
        EzTree { intervals }
    }

    fn set_max(v: &mut Vec<Node<T>>, i: usize) -> Option<u32> {
        if i >= v.len() {
            return None;
        }
        let left = Self::set_max(v, 2 * i + 1);
        let right = Self::set_max(v, 2 * i + 2);
        let node = &mut v[i];
        let left_right_max = match (left, right) {
            (Some(l), Some(r)) => Some(std::cmp::max(l, r)),
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        };
        let max = if let Some(l_r_max) = left_right_max {
            std::cmp::max(l_r_max, node.interval.stop)
        } else {
            node.interval.stop
        };
        node.max = max;
        Some(max)
    }

    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    // Insert items from the sorted iterator `iter` into `v` in complete binary tree order.
    //
    // Requires `iter` to be a sorted iterator. Requires v's capacity to be set to the number of
    // elements in `iter`. The length of `v` will not be changed by this function.
    // Find the first overlap position
    // TODO: See cgranges and how the stack implemenatino is used
    pub fn find<'a>(&'a self, start: u32, stop: u32) -> Vec<&'a Interval<T>> {
        // The finiky bit
        //
        // We want to prefetch a couple of levels down in the tree from where we are. However, we
        // can only fetch one cacheline at a time (assume a line holds 64b). We therefor need to
        // find at what depth a single prefetch fetches all the descendants. It turns out that, at
        // depth k under some node with index i, the leftmost child is at:
        //
        // 2^k * i + 2^(k-1) + 2^(k-2) + ... 2^0 = 2^k * i + 2^k - 1
        //
        // This follows from the fact that the leftmost immediate child of node i is at 2i + 1 by
        // recursivly expanding i. If your're curious the rightmost child is at:
        //
        // 2^k * i + 2^(k-1) + 2^(k-2) + ... + 2^0 = 2^k * i + 2^(k+1) - 1
        //
        // at depth k, there are 2^k children. We can fit 64/(sizeof(Interval<T>)) children in a
        // cacheline, so we want to use the depth k that is 64/sizeof(Interval<T>) children. So, we
        // want:
        //
        // 2^k = 64/sizeof(Interval<T>)
        //
        // But we don't actually *need* k, we only ever use 2^k. So, we can just use
        // 64/sizeof(Interval<T>) directly! We call this the multiplier.
        // now for those additions we had to do above. Well, we know that the offset is really just
        // 2^k = 1, and we know that multiplier == 2^k, so we're done right? Well sort of. The
        // prefetch instruction fetches the cacheline that *holds* the given memory address. Let's
        // denote cachelines with []. What if we have:
        //
        // [..., 2^k + 2^k-1] [2^k + s^k, ...]
        //
        // Essentially, we got unlucky with the alignment so that the leftmost child is not sharing
        // a cacheline with any of the other items at that level! That's not great. So, instead, we
        // prefetch the address that is half-way through the set of children. That way, we ensure
        // that we prefetch at least half of the items.

        let mut stack = std::collections::VecDeque::new();
        stack.push_back(StackCell {
            node: &self.intervals[0],
            index: 0,
            //tree_level: 0,
            left_checked: false,
        });
        let mut result = vec![];
        while let Some(mut stack_cell) = stack.pop_front() {
            #[cfg(feature = "nightly")]
            // unsafe is safe because pointer is never dereferenced
            unsafe {
                use std::intrinsics::prefetch_read_data;
                prefetch_read_data(
                    self.intervals
                        .as_ptr()
                        .offset((self.multiplier * stack_cell.index + self.offset) as isize),
                    3,
                )
            };
            // if left child not processed
            if !stack_cell.left_checked {
                let left_idx = 2 * stack_cell.index + 1;
                let left = self.intervals.get(2 * stack_cell.index + 1);
                // put current cell back on the stack
                stack_cell.left_checked = true;
                stack.push_back(stack_cell);
                // push left child if it exists and has chance of overlap
                if let Some(left) = left {
                    if left.max > start {
                        stack.push_back(StackCell {
                            node: left,
                            index: left_idx,
                            left_checked: false,
                        });
                    }
                }
            } else if let Some(right) = self.intervals.get(2 * stack_cell.index + 2) {
                // maybe push right onto stack
                if stack_cell.node.interval.start < stop {
                    // check if current node overlaps
                    if start < stack_cell.node.interval.stop {
                        //if node.interval.overlap(start, stop) {
                        result.push(&stack_cell.node.interval);
                    }
                    stack.push_back(StackCell {
                        node: right,
                        index: 2 * stack_cell.index + 2,
                        left_checked: false,
                    });
                }
            } else {
                if stack_cell.node.interval.overlap(start, stop) {
                    result.push(&stack_cell.node.interval);
                }
            }
        }
        result
    }
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;
    use ginterval::Interval;
    type Iv = Interval<u32>;
    type Lapper<T> = EzTree<T>;
    fn setup_nonoverlapping() -> Lapper<u32> {
        let data: Vec<Iv> = (0..100)
            .step_by(20)
            .map(|x| Iv {
                start: x,
                stop: x + 10,
                val: 0,
            })
            .collect();
        let lapper = Lapper::new(data);
        lapper
    }
    fn setup_overlapping() -> Lapper<u32> {
        let data: Vec<Iv> = (0..100)
            .step_by(10)
            .map(|x| Iv {
                start: x,
                stop: x + 15,
                val: 0,
            })
            .collect();
        let lapper = Lapper::new(data);
        lapper
    }
    fn setup_badlapper() -> Lapper<u32> {
        let data: Vec<Iv> = vec![
            Iv{start: 70, stop: 120, val: 0}, // max_len = 50
            Iv{start: 10, stop: 15, val: 0},
            Iv{start: 10, stop: 15, val: 0}, // exact overlap
            Iv{start: 12, stop: 15, val: 0}, // inner overlap
            Iv{start: 14, stop: 16, val: 0}, // overlap end
            Iv{start: 40, stop: 45, val: 0},
            Iv{start: 50, stop: 55, val: 0},
            Iv{start: 60, stop: 65, val: 0},
            Iv{start: 68, stop: 71, val: 0}, // overlap start
            Iv{start: 70, stop: 75, val: 0},
        ];
        let lapper = Lapper::new(data);
        lapper
    }
    fn setup_single() -> Lapper<u32> {
        let data: Vec<Iv> = vec![Iv {
            start: 10,
            stop: 35,
            val: 0,
        }];
        let lapper = Lapper::new(data);
        lapper
    }

    // Test that a query stop that hits an interval start returns no interval
    #[test]
    fn test_query_stop_interval_start() {
        let lapper = setup_nonoverlapping();
        let mut cursor = 0;
        assert_eq!(None, lapper.find(15, 20).get(0));
    }

    // Test that a query start that hits an interval end returns no interval
    #[test]
    fn test_query_start_interval_stop() {
        let lapper = setup_nonoverlapping();
        let mut cursor = 0;
        assert_eq!(None, lapper.find(30, 35).get(0));
    }

    // Test that a query that overlaps the start of an interval returns that interval
    #[test]
    fn test_query_overlaps_interval_start() {
        let lapper = setup_nonoverlapping();
        let mut cursor = 0;
        let expected = Iv {
            start: 20,
            stop: 30,
            val: 0,
        };
        assert_eq!(&expected, lapper.find(15, 25)[0]);
    }

    // Test that a query that overlaps the stop of an interval returns that interval
    #[test]
    fn test_query_overlaps_interval_stop() {
        let lapper = setup_nonoverlapping();
        let mut cursor = 0;
        let expected = Iv {
            start: 20,
            stop: 30,
            val: 0,
        };
        assert_eq!(&expected, lapper.find(25, 35)[0]);
    }

    // Test that a query that is enveloped by interval returns interval
    #[test]
    fn test_interval_envelops_query() {
        let lapper = setup_nonoverlapping();
        let mut cursor = 0;
        let expected = Iv {
            start: 20,
            stop: 30,
            val: 0,
        };
        assert_eq!(&expected, lapper.find(22, 27)[0]);
    }

    // Test that a query that envolops an interval returns that interval
    #[test]
    fn test_query_envolops_interval() {
        let lapper = setup_nonoverlapping();
        let mut cursor = 0;
        let expected = Iv {
            start: 20,
            stop: 30,
            val: 0,
        };
        assert_eq!(&expected, lapper.find(15, 35)[0]);
    }

    #[test]
    fn test_overlapping_intervals() {
        let lapper = setup_overlapping();
        let mut cursor = 0;
        let e1 = Iv {
            start: 0,
            stop: 15,
            val: 0,
        };
        let e2 = Iv {
            start: 10,
            stop: 25,
            val: 0,
        };
        let mut found = lapper.find(8, 20);
        found.sort();
        assert_eq!(vec![&e1, &e2], found);
    }

    //#[test]
    //fn test_merge_overlaps() {
        //let lapper = setup_badlapper();
        //let expected: Vec<&Iv> = vec![
            //&Iv{start: 10, stop: 16, val: 0},
            //&Iv{start: 40, stop: 45, val: 0},
            //&Iv{start: 50, stop: 55, val: 0},
            //&Iv{start: 60, stop: 65, val: 0},
            //&Iv{start: 68, stop: 120, val: 0}, // max_len = 50
        //];
        //let new_lapper = lapper.merge_overlaps();
        //assert_eq!(expected, new_lapper.iter().collect::<Vec<&Iv>>());
        
    //}

    #[test]
    fn test_interval_intersects() {
        let i1 = Iv{start: 70, stop: 120, val: 0}; // max_len = 50
        let i2 = Iv{start: 10, stop: 15, val: 0};
        let i3 = Iv{start: 10, stop: 15, val: 0}; // exact overlap
        let i4 = Iv{start: 12, stop: 15, val: 0}; // inner overlap
        let i5 = Iv{start: 14, stop: 16, val: 0}; // overlap end
        let i6 = Iv{start: 40, stop: 50, val: 0};
        let i7 = Iv{start: 50, stop: 55, val: 0};
        let i_8 = Iv{start: 60, stop: 65, val: 0};
        let i9 = Iv{start: 68, stop: 71, val: 0}; // overlap start
        let i10 = Iv{start: 70, stop: 75, val: 0};

        assert_eq!(i2.intersect(&i3), 5); // exact match
        assert_eq!(i2.intersect(&i4), 3); // inner intersect
        assert_eq!(i2.intersect(&i5), 1); // end intersect
        assert_eq!(i9.intersect(&i10), 1); // start intersect
        assert_eq!(i7.intersect(&i_8), 0); // no intersect
        assert_eq!(i6.intersect(&i7), 0); // no intersect stop = start
        assert_eq!(i1.intersect(&i10), 5); // inner intersect at start
    }


    #[test]
    fn test_find_overlaps_in_large_intervals() {
        let data1: Vec<Iv> = vec![
            Iv{start: 0, stop: 8, val: 0},
            Iv{start: 1, stop: 10, val: 0}, 
            Iv{start: 2, stop: 5, val: 0}, 
            Iv{start: 3, stop: 8, val: 0},
            Iv{start: 4, stop: 7, val: 0},
            Iv{start: 5, stop: 8, val: 0},
            Iv{start: 8, stop: 8, val: 0},
            Iv{start: 9, stop: 11, val: 0},
            Iv{start: 10, stop: 13, val: 0},
            Iv{start: 100, stop: 200, val: 0},
            Iv{start: 110, stop: 120, val: 0},
            Iv{start: 110, stop: 124, val: 0},
            Iv{start: 111, stop: 160, val: 0},
            Iv{start: 150, stop: 200, val: 0},
        ];
        let lapper = Lapper::new(data1);
        let mut found2 = lapper.find(8, 11);
        found2.sort();
        assert_eq!(found2, vec![
            &Iv{start: 1, stop: 10, val: 0}, 
            &Iv{start: 9, stop: 11, val: 0},
            &Iv{start: 10, stop: 13, val: 0},
        ]);
        let mut found2 = lapper.find(145, 151);
        found2.sort();
        assert_eq!(found2, vec![
            &Iv{start: 100, stop: 200, val: 0},
            &Iv{start: 111, stop: 160, val: 0},
            &Iv{start: 150, stop: 200, val: 0},
        ]);
    }

  

    
    // BUG TESTS - these are tests that came from real life

    // Test that it's not possible to induce index out of bounds by pushing the cursor past the end
    // of the lapper.
    //#[test]
    //fn test_seek_over_len() {
        //let lapper = setup_nonoverlapping();
        //let single = setup_single();
        //let mut cursor: usize = 0;

        //for interval in lapper.iter() {
            //for o_interval in single.seek(interval.start, interval.stop, &mut cursor) {
                //println!("{:#?}", o_interval);
            //}
        //}
    //}

    // Test that if lower_bound puts us before the first match, we still return a match
    #[test]
    fn test_find_over_behind_first_match() {
        let lapper = setup_badlapper();
        let e1 = Iv {start: 50, stop: 55, val: 0};
        let found2 = lapper.find(50, 55)[0];
        assert_eq!(found2, &e1);
    }

    // Test that seeking for all intervals over self == len(self)
    //#[test]
    //fn test_seek_over_nonoverlapping() {
        //let lapper = setup_nonoverlapping();
        //let mut total = 0;
        //let mut cursor: usize = 0;
        //for iv in lapper.iter() {
            //for _ in lapper.seek(iv.start, iv.stop, &mut cursor) {
                //total += 1;
            //}
        //}
        //assert_eq!(lapper.len(), total);
    //}

    // Test that finding for all intervals over self == len(self)
    #[test]
    fn test_find_over_nonoverlapping() {
        let lapper = setup_nonoverlapping();
        let mut total = 0;
        for iv in lapper.intervals.iter() {
            for _ in lapper.find(iv.interval.start, iv.interval.stop) {
                total += 1;
            }
        }
        assert_eq!(lapper.len(), total);
        total = 0;
        for iv in lapper.intervals.iter() {
            total +=  lapper.find(iv.interval.start, iv.interval.stop).len();
        }
        assert_eq!(lapper.len(), total);
    }

    // When there is a very long interval that spans many little intervals, test that the little
    // intevals still get returne properly
    #[test]
    fn test_bad_skips() {
        let data = vec![
            Iv{start:25264912, stop: 25264986, val: 0},	
            Iv{start:27273024, stop: 27273065	, val: 0},
            Iv{start:27440273, stop: 27440318	, val: 0},
            Iv{start:27488033, stop: 27488125	, val: 0},
            Iv{start:27938410, stop: 27938470	, val: 0},
            Iv{start:27959118, stop: 27959171	, val: 0},
            Iv{start:28866309, stop: 33141404	, val: 0},
        ];
        let lapper = Lapper::new(data);

        let found1 = lapper.find(28974798, 33141355);
        assert_eq!(found1, vec![
            &Iv{start:28866309, stop: 33141404	, val: 0},
        ]);
    }
}
