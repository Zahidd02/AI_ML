import pyautogui
from joblib import load
from tkinter import *
from skimage.io import imread
from skimage.transform import resize

# Run the main file to create the joblib first...
# Load Joblib model
model = load('Handwritten-digit_Recognition.joblib')

# Creating a 500x500 app with canvas
app = Tk(className=" Digit Recognizer")
app.geometry("500x500")
canvas = Canvas(app, bg='black')
canvas.pack(fill='both', expand=1)

# Creating label for the text box
label = Label(app, text="Predicted text")
label.config(font=("Courier", 14))
label.pack()

# Creating text box
text = Text(app, height=1, width=1, font=("Helvetica", 19))
text.tag_config("None", justify='center')
text.pack()


def click(click_event):
    global prev  # Stores first click and rest of X and Y movement co-ordinates
    prev = click_event


def move(move_event):
    global prev
    canvas.create_line(prev.x, prev.y, move_event.x, move_event.y, fill='white',
                       width=25, capstyle=ROUND, joinstyle=ROUND)  # Creates a brush that records movement in 'prev'
    prev = move_event


# Clears screen
def clear_screen():
    canvas.delete('all')
    text.delete("1.0", "end")


# Caller method
def find_digit():
    takeScreenshot()
    predict_digit()


# Digit resize and prediction using model.predict()
def predict_digit():
    img = imread('screenshot.png', as_gray=True)
    img = resize(img, (28, 28), anti_aliasing=True)
    img = img.reshape(-1, 784)
    img = img * 255
    img = img.astype(int)
    for x in range(0, 784):
        if img[0][x] <= 35:
            img[0][x] = 0
    value = model.predict(img)
    value = value[0]
    text.insert(END, value)


# Takes the temporary screenshot of the canvas
def takeScreenshot():
    # get the region of the canvas
    x, y = canvas.winfo_rootx(), canvas.winfo_rooty()
    w, h = canvas.winfo_width(), canvas.winfo_height()
    pyautogui.screenshot('screenshot.png', region=(x, y, w, h))


# Binds the left button to click() and move() methods
canvas.bind('<Button-1>', click)
canvas.bind('<B1-Motion>', move)

# Creating buttons
clear_btn = Button(app, text="Clear Screen", command=clear_screen)
find_num = Button(app, text="Find Digit", command=find_digit)

# Specifying button positions
find_num.pack(side=LEFT)
clear_btn.pack(side=RIGHT)

# Activate tkinter app
app.mainloop()
