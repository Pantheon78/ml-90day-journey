# Create the file with all your learnings
cat > day_01_foundations.md << 'EOF'
# Day 1: Vectors, Basis, and Linear Intuition
*Date: $(date)*

## 1. What I Learned Today
- Vectors as fundamental "moves" in space (like "East" = [1,0])
- A **basis** is a complete, non-redundant set of moves to reach any point
- The set {[1,0], [1,1]} is defective because it can't reach points like [0,1]

## 2. Teaching Snapshot (Explain Like I'm 10)
"A basis is like having only two magic buttons: 'Right' and 'Up'. With these, I can get anywhere on a grid. If I replace 'Up' with 'Up-and-Right', I can't move straight up anymore—some spots become impossible. That's why {[1,0], [1,1]} is a broken toolkit."

## 3. Code Proof
```python
import numpy as np
east = np.array([1, 0])
north = np.array([0, 1])
northeast = np.array([1, 1])

# Working combination
print("Reaching (5,-2):", 5*east + (-2)*north)

# Defective combination challenge
# Can't reach [0,1] with east and northeast!
