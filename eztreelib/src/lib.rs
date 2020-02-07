use ginterval::Interval;

pub struct EzTree<T: Default> {
    // FIXME: make not public and fix test that relies on this
    pub intervals: Vec<Interval<T>>,
    #[cfg(feature = "nightly")]
    mask: usize,
}

// Heavily based off of https://github.com/jonhoo/ordsearch/blob/master/src/lib.rs
impl<T: Default> EzTree<T> {
    pub fn new(mut intervals: Vec<Interval<T>>) -> Self {
        intervals.sort();
        let n = intervals.len();
        let mut iter = intervals.into_iter();
        let mut v = Vec::with_capacity(n);
        Self::eytzinger_walk(&mut v, &mut iter, 0);

        // it's now safe to set the length, since all `n` elements have been inserted.
        unsafe { v.set_len(n) };

        // Maybe nightly only
        #[cfg(feature = "nightly")]
        {
            let mut mask = 1;
            while mask <= n {
                mask <<= 1;
            }
            mask -= 1;
            EzTree {
                intervals: v,
                mask: mask,
            }
        }
        #[cfg(not(feature = "nightly"))]
        EzTree { intervals: v }
    }

    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    // Insert items from the sorted iterator `iter` into `v` in complete binary tree order.
    //
    // Requires `iter` to be a sorted iterator. Requires v's capacity to be set to the number of
    // elements in `iter`. The length of `v` will not be changed by this function.
    fn eytzinger_walk<I>(v: &mut Vec<Interval<T>>, iter: &mut I, i: usize)
    where
        I: Iterator<Item = Interval<T>>,
    {
        if i >= v.capacity() {
            return;
        }
        // visit left child
        Self::eytzinger_walk(v, iter, 2 * i + 1);

        // put data at the root
        // we know the get_unchecked_mut and unwrap below are safe bc we set the Vec's cap to the
        // len of te iterator.
        *unsafe { v.get_unchecked_mut(i) } = iter.next().unwrap();

        // vist right child
        Self::eytzinger_walk(v, iter, 2 * i + 2);
    }

    // Find the first overlap position
    pub fn find<'a>(&'a self, start: u32, stop: u32) -> Vec<&'a Interval<T>> {
        let mut result = vec![];
        use std::mem;
        let mut i = 0;

        // The finiky bit
        //
        // We want to prefetch a couple of levels down in the tree from where we are. However, we
        // can only fetch one cacheline at a time (assume a line holds 64b). We therefor need to
        // find at what depth a single prefetch fetches all the descendants. It turns out that, at
        // depth k under some node with index i, the leftmost child is at:
        //
        // 2^k * i + 2^(k-1) + 2^(k-2) + ... 2^0 = s^k * i + s^k - 1
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
        let multiplier = 64 / mem::size_of::<Interval<T>>();
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
        let offset = multiplier + multiplier / 2;
        let _ = offset; // make nightly happy if we use it

        while i < self.intervals.len() {
            #[cfg(feature = "nightly")]
            // unsafe is safe because pointer is never dereferenced
            unsafe {
                use std::intrinsics::prefetch_read_data;
                prefetch_read_data(
                    self.items
                        .as_ptr()
                        .offset(((multiplier * i + offset) & self.mask) as isize),
                    3,
                )
            };

            // safe because i < self.intervals.len()
            let interval = unsafe { self.intervals.get_unchecked(i) };
            if interval.overlap(start, stop) {
                result.push(interval);
                i = 2 * i + 1;
            } else if interval.stop > start {
                break;
            } else {
                i = 2 * i + 2;
            }
        }

        // we want ffs(~(i + 1))
        // since ctz(x) = ffs(x) - 1
        // we use ctz(~(i + 1)) + 1
        //let j = (i + 1) >> ((!(i + 1)).trailing_zeros() + 1);
        //if j != 0 {
        //result.push(unsafe { self.intervals.get_unchecked(j - 1) });
        //}
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
        assert_eq!(vec![&e1, &e2], lapper.find(8, 20));
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
        let found2 = lapper.find(8, 11);
        assert_eq!(found2, vec![
            &Iv{start: 1, stop: 10, val: 0}, 
            &Iv{start: 9, stop: 11, val: 0},
            &Iv{start: 10, stop: 13, val: 0},
        ]);
        let found2 = lapper.find(145, 151);
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
            for _ in lapper.find(iv.start, iv.stop) {
                total += 1;
            }
        }
        assert_eq!(lapper.len(), total);
        total = 0;
        for iv in lapper.intervals.iter() {
            total +=  lapper.find(iv.start, iv.stop).len();
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
