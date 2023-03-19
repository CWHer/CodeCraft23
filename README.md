# CodeCraft23

## Hierarchical Control

**task**: x {BUY, SELL, DEL} y at {z, None}

**sub-task**: x GOTO z, x {BUY, SELL, DEL} y

1. Decision: `observation` -(`scheduler`)-> `task`

2. Control: `task` -(`genSubtask`)-> `subtask` -(`genAction`)-> `action`

**What could be improve**

1. How does scheduler select tasks

2. How to detect conflict/deadlock between tasks

3. fine-grained action generation