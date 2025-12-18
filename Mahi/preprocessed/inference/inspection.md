
```python
with open("../digits_data_cleaned.pickle", "rb") as f:
  clean_data = pickle.load(f)

print(f"\nKeys found in the pickle file: {list(clean_data.keys())}")
X_train_clean = clean_data["X_train"]
y_train_clean = clean_data["y_train"]
X_val_clean = clean_data["X_val"]
y_val_clean = clean_data["y_val"]
categories_clean = clean_data["categories"]
print("\n--- Data Structure & Shapes ---")
print(f"X_train shape: {X_train_clean.shape}")
print(f"y_train shape: {y_train_cleean.shape}")
print(f"X_val shape: {X_val_clean.shape}")
print(f"y_val shape: {y_val_clean.shape}")
```

## output:
```
--- Data Structure & Shapes ---
X_train shape: (416126, 32, 32, 1)
y_train shape: (416126,)
X_val shape: (11453, 32, 32, 1)
y_val shape: (11453,)
```


```python
print("\n--- Shape Explanation ---")
print(f"For X_train {X_train_clean.shape}:")
print(f"  {X_train_clean.shape[0]} -> Number of images (Batch size)")
print(f"  {X_train_clean.shape[1]} -> Image Height (Pixels)")  
print(f"  {X_train_clean.shape[2]} -> Image Width (Pixels)")
print(f"  {X_train_clean.shape[3]} -> Color Channels (1 = Grayscale)")
```
## output:
```
--- Shape Explanation ---
For X_train (416126, 32, 32, 1):
  416126 -> Number of images (Batch size)
  32 -> Image Height (Pixels)
  32 -> Image Width (Pixels)
  1 -> Color Channels (1 = Grayscale)
```
  

```python
print("\n--- Content Inspection (First Training Image) ---")
print(f"Data Type: {X_train_clean.dtype}")
print(f"Min Value: {X_train_clean.min()}")
print(f"Max Value: {X_train_clean.max()}")
print(f"Label Index: {y_train_clean[0]}")
print(f"Label Name:  {categories_clean[y_train_clean[0]]}")
```
## output:
```
--- Content Inspection (First Training Image) ---
Data Type: uint8
Min Value: 0
Max Value: 255
Label Index: 0
Label Name:  0
```


```python
clean_grid = X_train_clean[0].reshape(32, 32)
np.set_printoptions(linewidth=200, edgeitems=4)
with np.printoptions(threshold=np.inf):
  print(clean_grid)
```
## output:
```
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  74  64   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 131 201 238 231 174  97   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0  63 172 242 253 255 254 253 200  55   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  91 202 249 253 254 238 193 226 228 141  68   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0  63 131 225 255 253 255 255 229 123 137 235 255 208  68   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0 106 221 249 253 255 232 237 253 253 195  87 179 253 225  88   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  73 189 254 255 248 194 111 177 240 205 114   0  88 223 231 107   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0  94 207 250 255 245 166  55   0  96 109   0   0   0   0 192 251 188  58   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0  83 197 255 239 194 185  98   0   0   0   0   0   0   0   0 198 251 244 139   0   0   0   0   0   0   0]
 [  0   0   0   0   0  98 223 251 231 131   0   0   0   0   0   0   0   0   0   0   0 195 251 255 169   0   0   0   0   0   0   0]
 [  0   0   0   0  55 195 249 226 109   0   0   0   0   0   0   0   0   0   0   0   0 193 255 254 167   0   0   0   0   0   0   0]
 [  0   0   0   0 144 242 243 153   0   0   0   0   0   0   0   0   0   0   0   0   0 197 255 255 168   0   0   0   0   0   0   0]
 [  0   0   0  81 218 255 225  82   0   0   0   0   0   0   0   0   0   0   0   0   0 193 253 236 130   0   0   0   0   0   0   0]
 [  0   0   0 103 233 249 164   0   0   0   0   0   0   0   0   0   0   0   0   0 135 237 236 123   0   0   0   0   0   0   0   0]
 [  0   0   0 106 239 248 137   0   0   0   0   0   0   0   0   0   0   0  54 161 233 220 152   0   0   0   0   0   0   0   0   0]
 [  0   0   0 101 231 226 103   0   0   0   0   0   0   0   0   0  52 115 196 231 164  62   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0 103 237 238 114   0   0   0   0   0   0   0   0 102 188 251 236 135   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0 101 234 249 208 101   0   0   0  67 124 177 210 241 236 206 152   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0 105 235 255 249 240 202 163 209 232 247 252 235 219 148   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0  63 186 251 251 255 253 254 254 254 240 203 110   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0  58 144 213 243 250 247 231 178 105   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0  73  78  78  64   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]]
```