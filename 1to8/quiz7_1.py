from tkinter import *
from tkinter.filedialog import *
from tkinter.simpledialog import *
import os.path
import math 


def loadImage(fname):
    global window, canvas, paper, filename, XSIZE, YSIZE, inImage
    
    inImage=[]
    fsize=os.path.getsize(fname)
    XSIZE=YSIZE=int(math.sqrt(fsize))

    fp=open(fname,'rb')

    for i in range(0,XSIZE):
        tmpList=[]
        for k in range(0,YSIZE):
            data=int(ord(fp.read(1)))
            tmpList.append(data)
        inImage.append(tmpList)

    fp.close()

def displayImage(image):
    global window, canvas, paper, filename, XSIZE, YSIZE, inImage
    rgbString=""
    for i in range(0,XSIZE):
        tmpString=""
        for k in range(0,YSIZE):
            data=image[i][k]
            tmpString+="#%02x%02x%02x " %(data,data,data)
        rgbString+="{"+tmpString+"} "
    paper.put(rgbString)

def func_open():
    global window, canvas, paper, filename, XSIZE, YSIZE, inImage

    filename=askopenfilename(parent=window,filetypes=(("RAW 파일","*.raw"),("모든 파일","*.*")))
    if filename =='':
        return
    
    #if canvas!=None:
    #    canvas.destory()

    

    loadImage(filename)

    window.geometry(str(XSIZE)+'x'+str(YSIZE))
    canvas=Canvas(window,height=XSIZE,width=YSIZE)
    paper=PhotoImage(width=XSIZE,height=YSIZE)
    canvas.create_image((XSIZE/2,YSIZE/2),image=paper,state="normal")


    displayImage(inImage)

    canvas.pack()

def func_exit():
    window.quit()
    window.destory()

def brightPhoto():
    global window, canvas, paper, filename, XSIZE, YSIZE, inImage
    value=0
    value=askinteger('밝게','값 입력',minvalue=1,maxvalue=255)

    for i in range(0,XSIZE):
        for k in range(0,YSIZE):
            data=inImage[i][k]+value
            if data>255:
                newData=255
            else:
                newData=data
            inImage[i][k]=newData

    displayImage(inImage)

def darkPhoto():
    global window, canvas, paper, filename, XSIZE, YSIZE, inImage
    value=0
    value=askinteger('어둡게','값 입력',minvalue=1,maxvalue=255)

    for i in range(0,XSIZE):
        for k in range(0,YSIZE):
            data=inImage[i][k]-value
            if data<0:
                newData=0
            else:
                newData=data
            inImage[i][k]=newData

    displayImage(inImage)

def reverserPhoto():
    global window, canvas, paper, filename, XSIZE, YSIZE, inImage

    for i in range(0,XSIZE):
        for k in range(0,YSIZE):
            data=inImage[i][k]
            newData=255-data
            inImage[i][k]=newData

    displayImage(inImage)


window=None
canvas=None
XSIZE,YSIZE=0,0
inImage=[]
filename=''


if __name__=="__main__":
    window=Tk()
    window.title("흑백 사진 보기")

    
    mainMenu=Menu(window)
    window.config(menu=mainMenu)
    fileMenu=Menu(mainMenu)
    mainMenu.add_cascade(label="파일",menu=fileMenu)
    fileMenu.add_command(label="파일 열기",command=func_open)
    fileMenu.add_separator()
    fileMenu.add_command(label="프로그램 종료",command=func_exit)

    photoMenu=Menu(mainMenu)
    mainMenu.add_cascade(label="사진효과",menu=photoMenu)
    photoMenu.add_command(label="밝게하기",command=brightPhoto)
    photoMenu.add_command(label="어둡게하기",command=darkPhoto)
    photoMenu.add_command(label="반전 이미지",command=reverserPhoto)

    window.mainloop()
