# CodeCraft23

## Hierarchical Control

**task**: x {BUY, SELL, DEL} y at {z, None}

**sub-task**: x GOTO z, x {BUY, SELL, DEL} y

1. Decision: `observation` -(`scheduler`)-> `task`

2. Control: `task` -(`genSubtask`)-> `subtask` -(`genAction`)-> `action`