import cv2
import numpy as np
import easyocr

# Load the image - Gorseli yukluyoruz. Proje genelinde resim islemlerini OpenCV ile yapiyoruz.
img = cv2.imread('resources/sudoku2.png')
# Convert the image to grayscale - Gorseli griye ceviriyoruz.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image - Gorseli bulaniklastiriyoruz.
blur = cv2.GaussianBlur(gray, (3, 3), 0)
# Threshold the image - Gorseli siyah beyaz yapiyoruz. Boylelikle hem sinirlarimizi daha iyi belirliyoruz hem de rakamlari daha iyi taniyabilecegiz.
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
# Find the contours - Gorseldeki sinirlari buluyoruz. 
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Find the biggest contour - Gorseldeki en buyuk siniri buluyoruz. Bu sinir sudoku tahtasini temsil ediyor ve en buyugu buldugumuz icin kalanlari bolmek icin ayarlamalar yapacagiz.
biggest = np.array([])
max_area = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > 200:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
cv2.drawContours(img, biggest, -1, (0, 255, 0), 3)
# Reorder the points in the contour - Gorseldeki sinir noktalarini yeniden duzenliyoruz. Boylelikle daha iyi bir perspektif donusumu yapabilecegiz.
biggest = biggest.reshape((4, 2))
new_biggest = np.zeros((4, 1, 2), dtype=np.int32)
add = biggest.sum(1)
new_biggest[0] = biggest[np.argmin(add)]
new_biggest[3] = biggest[np.argmax(add)]
diff = np.diff(biggest, axis=1)
new_biggest[1] = biggest[np.argmin(diff)]
new_biggest[2] = biggest[np.argmax(diff)]
biggest = new_biggest

# Create a new image with the biggest contour - Gorseldeki en buyuk siniri yeni bir gorselde gosteriyoruz.
pts1 = np.float32(biggest)
pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
img_warp = cv2.warpPerspective(img, matrix, (500, 500))

# Divide the warped image into 81 small cells - Gorseli 81 kucuk hucreye boluyoruz. Boylelikle her hucredeki rakami daha iyi taniyabilecegiz.
cell_size = 500 // 9
sudoku_board = np.zeros((9, 9), dtype=int)  # Initialize the Sudoku board with zeros - Sudoku tahtasini sifirlarla dolduruyoruz.

reader = easyocr.Reader(['en'], )  # Initialize EasyOCR with the desired language - EasyOCR'i istedigimiz dilde baslatiyoruz. Rakam taniyacagimiz icin bu kisim onemli degil. Turkce ya da ingilizce yazabilirsiniz. Biz daha garanti sonuclar elde etmek icin ingilizce yazdik.

# Read the digits in each cell - Her hucredeki rakamlari okuyoruz.
for i in range(9):
    for j in range(9):
        x = j * cell_size
        y = i * cell_size
        cell_img = img_warp[y:y+cell_size, x:x+cell_size]
        cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        cell_text = reader.readtext(cell_gray, detail=0, allowlist = '0123456789', mag_ratio=2)
        #En can alici nokta burasi oluyor. EasyOCR ile okudugumuz rakamlari sudoku tahtasina yerlestiriyoruz. Eger okudugumuz rakam bosluk ise 0 olarak yerlestiriyoruz.Cunku sudokuyu cozerken bosluklari kullanamayacagiz. 
        if len(cell_text) > 0 and cell_text[0].isdigit():
            sudoku_board[i][j] = int(cell_text[0])
        else:
            sudoku_board[i][j] = 0

# Print the Sudoku board - Sudoku tahtasini yazdiriyoruz.
print(sudoku_board)




def solve_sudoku(board):
    # Find an empty cell on the board - Tahtadaki bosluklari buluyoruz.
    empty_cell = find_empty_cell(board)
    if not empty_cell:
        # If there are no empty cells left, the Sudoku is solved - Eger bosluk kalmadiysa sudoku cozulmustur.
        return True
    
    row, col = empty_cell

    for num in range(1, 10):
        if is_valid(board, num, row, col):
            # If the current number is valid, place it in the cell - Eger ki buldugumuz sayi gecerli bir sayiysa o sayiyi hucreye yerlestiriyoruz.
            board[row][col] = num

            if solve_sudoku(board):
                # Recursively solve the Sudoku - Sudoku'yu tekrar cagiriyoruz.
                return True

            # If the current placement leads to an invalid solution,
            # backtrack and try a different number - Eger ki buldugumuz sayi gecerli bir sayi degilse, sayiyi geri aliyoruz ve farkli bir sayi deniyoruz.
            board[row][col] = 0

    # If no numbers lead to a valid solution, return False - Eger ki hicbir sayi gecerli bir cozum uretmiyorsa False donduruyoruz.
    return False

def find_empty_cell(board):
    # Find the next empty cell on the board - Tahtadaki bir sonraki boslugu buluyoruz.
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_valid(board, num, row, col):
    # Check if placing a number in a cell is valid - Bir sayiyi hucreye yerlestirmenin gecerli olup olmadigini kontrol ediyoruz.
    return (
        is_valid_row(board, num, row) and
        is_valid_column(board, num, col) and
        is_valid_box(board, num, row - row % 3, col - col % 3)
    )

def is_valid_row(board, num, row):
    # Check if placing a number in a row is valid - Bir sayiyi satira yerlestirmenin gecerli olup olmadigini kontrol ediyoruz.
    for j in range(9):
        if board[row][j] == num:
            return False
    return True

def is_valid_column(board, num, col):
    # Check if placing a number in a column is valid - Bir sayiyi sutuna yerlestirmenin gecerli olup olmadigini kontrol ediyoruz.
    for i in range(9):
        if board[i][col] == num:
            return False
    return True

def is_valid_box(board, num, start_row, start_col):
    # Check if placing a number in a 3x3 box is valid - Bir sayiyi 3x3 kutucuga yerlestirmenin gecerli olup olmadigini kontrol ediyoruz.
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                return False
    return True

def print_sudoku_board(board):
    for i in range(9):
        for j in range(9):
            print(board[i][j], end=' ')
        print()

# Create a copy of the Sudoku board to preserve the original - Orijinal sudoku tahtasini korumak icin sudoku tahtasinin bir kopyasini olusturuyoruz.
sudoku_board_copy = sudoku_board.copy()

# Solve the Sudoku board - Sudoku tahtasini cozmeye basliyoruz.
if solve_sudoku(sudoku_board_copy):
    # If a solution is found, print the solved board to the console - Eger ki bir cozum bulunduysa, cozulmus sudoku tahtasini konsola yazdiriyoruz.
    print("Solved Sudoku:")
    print_sudoku_board(sudoku_board_copy)
else:
    print("No solution exists for the given Sudoku board.")

img_height, img_width, _ = img.shape

# Her hücrenin genişliği ve yüksekliği
cell_width = img_width // 9
cell_height = img_height // 9

# Bulmaca çözümünü içeren matrisi alın
solution_matrix = sudoku_board_copy


# Orijinal bulmaca fotoğrafının üzerine çözümü yazdır
for i in range(9):
    for j in range(9):
        # Hücrenin sol üst köşesinin koordinatları
        x = j * cell_width
        y = i * cell_height

        # Hücreye yazılacak sayı
        number = solution_matrix[i][j]

        # Mevcut sayıyı kontrol et
        if number != 0:
            # Sayıyı fotoğrafın üzerine yazdır
            text_size, _ = cv2.getTextSize(str(number), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = x + (cell_width - text_size[0]) // 2
            text_y = y + (cell_height + text_size[1]) // 2
            cv2.putText(img, str(number), (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

# Sonucu görüntüle
cv2.imshow('Solved Sudoku', img)
cv2.waitKey(0)
cv2.destroyAllWindows()