#coding = utf-8 
from tkinter import * 
from tkinter import StringVar 
import time 
import math

class Progress():

    def __init__(self, title, on=False):
        self.on=on
        if on is False:
            return 

        else:
            self.root = Tk() 
            self.root.geometry('300x100') 
            self.root.title(title) 
            self.var = StringVar()
            self.var.set("") 
            self.button = Button(self.root,textvariable = self.var, width = 5) 
            self.button.grid(row = 0,column = 0,padx = 5)
            self.canvas = Canvas(self.root,width = 170,height = 26,bg = "white")        
            self.out_line = self.canvas.create_rectangle(2,2,180,27,width = 1,outline = "black") 
            self.canvas.grid(row = 0,column = 1,ipadx = 5) 

            self.estimate_time_label = Label(self.root, text = 'Elapsed Time:')
            self.estimate_time_label.grid(row = 1, column = 0 )

            self.estimate_time_text = StringVar()
            self.estimate_time_text.set('00:00:00')

            self.estimate_time = Label(self.root, textvariable = self.estimate_time_text)
            self.estimate_time.grid( row = 1, column = 1 )

            self.remain_time_label = Label(self.root, text = 'Remain Time:')
            self.remain_time_label.grid(row = 2, column = 0 )

            self.remain_time_text = StringVar()
            self.remain_time_text.set('00:00:00')

            self.remain_time = Label(self.root, textvariable = self.remain_time_text)
            self.remain_time.grid( row = 2, column = 1 )
            #self.root.mainloop() 
            
            self.time=time.time()
            self.init_time=time.time()    

    def log(self, progress):
        if self.on:
            n = 1.8*progress*100
            fill_line = self.canvas.create_rectangle(2,2,0,27,width = 0,fill = "blue") 
            self.canvas.coords(fill_line, (0, 0, n, 30)) 
            if progress*100 >= 100: 
                self.var.set("100%") 
            else: 
                self.var.set(str(round(progress*100,1))+"%")   

            self.time=time.time()

            def get_hms(time_):
                m, s = divmod(time_, 60)
                h, m = divmod(m, 60)
                h=math.floor(h)
                m=math.floor(m)
                s=math.floor(s)
                return h,m,s

            elpased = self.time-self.init_time    
            h,m,s=get_hms(elpased)
            self.estimate_time_text.set('{:d}:{:d}:{:d}'.format(h,m,s))
            if progress==0:
                return
            
            remain = elpased/progress*(1-progress)
            h,m,s=get_hms(remain)
            self.remain_time_text.set('{:d}:{:d}:{:d}'.format(h,m,s))

            self.root.update() 
            
    def destroy(self):
        if self.on:
            self.root.destroy()

if __name__ == '__main__': 
    bar = Progress('test1')   
    for i in range(100):
        bar.log(i+1)

        bar2 = Progress('test2')
        for j in range(100):
            bar2.log(j+1)

        bar2.destroy()

    bar.destroy()