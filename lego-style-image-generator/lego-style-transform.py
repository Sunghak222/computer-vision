import cv2  
from PIL import Image, ImageDraw
import numpy as np
import os
import time

BRICK_LEN = 9                   # length of one lego
MAX_BRICK = 100                 # maximum number of bricks
LINE_COLOR = (100,100,100)      # color of lines
LINE_WIDTH = 1                  # width of boundaries of legos
SHADOW_COLOR = (100,100,100)    # color of shadows
LEGO_SHAPES = [[4,2], [2,4], [2,2], [2,1], [1,2], [1,1]]
input_h = input_w = scale = grid_h = grid_w = out_h = out_w = 0 # for global vars


# 1/3 of pixel value -> black, 2/3 of it -> gray, rest of it -> white
def quantize_color(input, n):

    # n==1 will lead to divide by zero error.
    out = np.zeros((input.shape[0],input.shape[1],3), dtype=np.uint8)
    if n == 1:
        return out

    colors = []
    step = 256/n

    # compute colors to be quantized
    for i in range(n):
        colors.append(int(255/(n-1)*i))
        # print(colors[i])

    # quantize input image
    for i in range(n):
        color = colors[i]
        lower_bound = step*i
        upper_bound = step*(i+1)
        # print(f"lower: {lower_bound}, upper: {upper_bound}")
        out[(input >= lower_bound) & (input < upper_bound)] = (color,color,color)

    return out

def make_grid(input, max_brick=MAX_BRICK):
    input_w, input_h = input.size
    scale = min(max_brick/input_h, max_brick/input_w, 1.0)
    grid_h = max(1, int(input_h*scale))
    grid_w = max(1, int(input_w*scale))

    # NEAREST, BILINEAR, BICUBIC, LANCZOS are tested
    grid = input.resize((grid_w, grid_h), Image.Resampling.NEAREST)
    return grid


def get_stud_color(color):
    r, g, b = map(int, color)
    c_avg = (r+g+b)//3

    # if brighter than average, make stud darker; if darker than average, make stud brighter
    if c_avg >= 128:
        delta = 0.95
    else:
        delta = 1.1

    r = min(max(int(r * delta), 0), 255)
    g = min(max(int(g * delta), 0), 255)
    b = min(max(int(b * delta), 0), 255)

    return (r,g,b)

def draw_one_grid(draw, x0, y0, brick_len, color, line_width=1):
    x1 = x0+BRICK_LEN - 1
    y1 = y0+BRICK_LEN - 1
    draw.rectangle([x0,y0,x1,y1], fill = color)

    cx = (x0 + x1 + LINE_WIDTH) // 2
    cy = (y0 + y1 + LINE_WIDTH) // 2
    r = int(BRICK_LEN * 0.3)
    stud_color = get_stud_color(color)

    # shadow should be vertically larger than stud for it to be visible
    draw.ellipse([cx-r, cy-(r*0.8), cx+r, cy+(r*1.5)], fill=SHADOW_COLOR)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=stud_color)

def draw_1x1_lego(input, brick_len=BRICK_LEN, line_width=LINE_WIDTH):
    np_input = np.array(input, dtype=np.uint8)
    grid_w, grid_h = input.size
    out_h = grid_h * BRICK_LEN
    out_w = grid_w * BRICK_LEN

    out = Image.new("RGB", (out_w,out_h), (0,0,0))
    draw = ImageDraw.Draw(out)
    grid_w, grid_h = input.size

    for y in range(grid_h):
        for x in range(grid_w):
            y0 = y*BRICK_LEN
            x0 = x*BRICK_LEN

            color = tuple(np_input[y][x])

            draw_one_grid(draw, x0, y0, brick_len, color, line_width)

    return out

def draw_boundaries(out, grid_w, grid_h, brick_len=BRICK_LEN,
                    line_color=LINE_COLOR, line_width=LINE_WIDTH): 
     
    draw = ImageDraw.Draw(out)  

    out_w = grid_w * brick_len  
    out_h = grid_h * brick_len  

    for x in range(1, grid_w):  
        x0 = x * brick_len      
        draw.line([(x0, 0), (x0, out_h)], fill=line_color, width=line_width)  

    for y in range(1, grid_h):  
        y0 = y * brick_len      
        draw.line([(0, y0), (out_w, y0)], fill=line_color, width=line_width)  

    return out

def get_color_distance(color1, color2):
    # Manhattan distance
    return abs(int(color1[0]) - int(color2[0])) + abs(int(color1[1]) - int(color2[1])) + abs(int(color1[2]) - int(color2[2]))

def are_same_color(color1, color2, threshold):
    # return if color1 and color2 are similar enough based on the threshold
    return get_color_distance(color1, color2) <= threshold

def are_same_color_region(grid: np.ndarray, x, y, w, h, threshold):
    # check if all pixels in the region [y:y+h][x:x+w] are similar enough to the color of (x,y)
    color = grid[y][x]
    for i in range(y,y+h):
        for j in range(x,x+w):
            if not are_same_color(color, grid[i][j], threshold):
                return False
    return True

def region_mean_color(grid: np.ndarray, x: int, y: int, w: int, h: int):
    # Return mean RGB of a brick region
    region = grid[y:y+h, x:x+w]
    r = int(region[:, :, 0].mean())
    g = int(region[:, :, 1].mean())
    b = int(region[:, :, 2].mean())
    return (r, g, b)

def can_place(grid, occupied, x, y, dx, dy, t):
    H, W = grid.shape[0], grid.shape[1]

    # boundary check
    if x < 0 or y < 0 or x + dx > W or y + dy > H:
        return False

    # must be empty and color similar to (x,y)
    color = grid[y, x]
    for yy in range(y, y + dy):
        for xx in range(x, x + dx):
            # check if the current pixel is already occupied
            if occupied[yy][xx]:
                return False
            # color similarity
            if not are_same_color(color, grid[yy,xx], t):
                return False

    return True

def mark_occupied(occupied, x, y, dx, dy):
    # mark [y:y+dy][x:x+dx] as True
    for yy in range(y, y + dy):
        for xx in range(x, x + dx):
            occupied[yy][xx] = True

def greedy_tiling(grid, t, brick_types=LEGO_SHAPES):
    H, W = grid.shape[0], grid.shape[1]
    occupied = [[False for _ in range(W)] for _ in range(H)]

    bricks = []  # list of placed bricks
    counts = {}  # summary counts, key like "2x4". (e.g. (2x4) : 3)

    for y in range(H):
        for x in range(W):
            if occupied[y][x]:
                continue

            # try bigger blocks first
            placed = False
            for (dy, dx) in brick_types:
                if can_place(grid, occupied, x, y, dx, dy, t):
                    # choose a mean color
                    c = region_mean_color(grid, x, y, dx, dy)
                    bricks.append((x, y, dx, dy, c))         

                    # ex) 2x4: 3
                    key = str(dx) + "x" + str(dy)           
                    counts[key] = counts.get(key, 0) + 1    

                    mark_occupied(occupied, x, y, dx, dy)   
                    placed = True
                    break

    return bricks, counts

def render_bricks(bricks, grid_w, grid_h, brick_len):
    out_w = grid_w * brick_len
    out_h = grid_h * brick_len

    out = Image.new("RGB", (out_w, out_h), (100,100,100))
    draw = ImageDraw.Draw(out)

    for (x, y, dx, dy, color) in bricks:
        x0 = x * brick_len
        y0 = y * brick_len
        x1 = (x + dx) * brick_len - 1 # -1 because PIL counts last pixel 
        y1 = (y + dy) * brick_len - 1

        # brick body with gap border
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=LINE_COLOR, width=LINE_WIDTH)

        # studs: dy rows * dx cols
        r = int(brick_len * 0.3)
        for i in range(dy):
            for j in range(dx):
                cx = (x0 + j*brick_len) + brick_len//2
                cy = (y0 + i*brick_len) + brick_len//2

                stud_color = get_stud_color(color)
                shadow_color = SHADOW_COLOR

                # base shadow
                # draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=shadow_color)
                draw.ellipse([cx-r, cy-(r*0.8), cx+r, cy+(r*1.5)], fill=shadow_color)

                # stud slightly up for highlight effect
                # draw.ellipse([cx-r, cy-r-1, cx+r, cy+r-1], fill=stud_color)
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=stud_color)

    return out

def pil_to_cv(img_pil):
    # PIL(RGB) -> OpenCV(BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(frame_bgr):
    # OpenCV(BGR) -> PIL(RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def main():

    # Try multiple camera indices (0,1,2) to find a working camera
    cam_idx = 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        cam_idx = 1
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        cam_idx = 2
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Failed to open camera. Try changing cam_idx or check camera permission.")
        return

    out_dir = "captures"
    os.makedirs(out_dir, exist_ok=True)

    print("Press 's' to save a snapshot, 'q' to quit.")
    print("Press '1' for Task2 mode (1x1 + 3 colors), '3' for Task3 mode (multi-size).")

    # default: Task3 (multi brick), press '1' for Task2 (1x1 brick), '3' for Task3 (multi shapes of brick and multi colors)
    mode = 3

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        # ---------- Original view ----------
        cv2.imshow("Task4 - Original Camera", frame)

        # ---------- Convert frame -> PIL ----------
        frame_pil = cv_to_pil(frame)

        # ---------- Build grid (<=100x100) ----------
        grid_pil = make_grid(frame_pil)       # PIL image, small (grid_w, grid_h)

        # ---------- Render LEGO ----------
        if mode == 1:
            # Task2 style: grayscale + 3 colors + 1x1 bricks
            grid_np = np.array(grid_pil)
            grid_gray = cv2.cvtColor(grid_np, cv2.COLOR_RGB2GRAY)
            qt = quantize_color(grid_gray, 3)                # (H,W,3) grayscale RGB
            qt_pil = Image.fromarray(qt.astype(np.uint8))
            grid_qt = make_grid(qt_pil)                      # ensure <=100x100

            out_pil = draw_1x1_lego(grid_qt, BRICK_LEN, LINE_WIDTH)
            grid_w, grid_h = grid_qt.size
            draw_boundaries(out_pil, grid_w, grid_h, BRICK_LEN, LINE_COLOR, LINE_WIDTH)

        else:
            # Task3 style: multi-size greedy tiling on color grid
            grid_np = np.array(grid_pil)                     # (H,W,3) RGB
            bricks, counts = greedy_tiling(grid_np, t=70)    # t can be tuned
            out_pil = render_bricks(
                bricks,
                grid_w=grid_np.shape[1],
                grid_h=grid_np.shape[0],
                brick_len=BRICK_LEN
            )
            
            # print summary counts
            total = 0
            for k in counts:
                total += counts[k]
            print("Total bricks:", total)
            for k in sorted(counts.keys()):
                print(k, ":", counts[k])

        # ---------- Show LEGO (PIL -> OpenCV) ----------
        out_cv = pil_to_cv(out_pil)
        cv2.imshow("Task4 - LEGO Output", out_cv)

        # ---------- Key handling ----------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path_raw = os.path.join(out_dir, f"raw_{ts}.png")
            path_lego = os.path.join(out_dir, f"lego_{ts}.png")
            cv2.imwrite(path_raw, frame)
            out_pil.save(path_lego)
            print("Saved:", path_raw, "and", path_lego)
        elif key == ord('1'):
            mode = 1
            print("Mode -> Task2 (1x1 + 3 colors)")
        elif key == ord('3'):
            mode = 3
            print("Mode -> Task3 (multi-size bricks)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()