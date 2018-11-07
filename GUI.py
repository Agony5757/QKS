from tkinter import *

root = Tk()
root.title("Quantum Kitchen Sinks")
root.geometry('1000x500')
root.resizable(width=False, height=False)

#frame
QKS_title=Frame(master=root)
QKS_option=Frame(master=root)
QKS_dataset_import=Frame(master=root)
QKS_output=Frame(master=root)
#title
title_label=Label(master=QKS_title,text='Quantum Kitchen Sinks')
title_label.pack()

#option
optionbox_=Entry(master=QKS_option)


#dataset import
dataset=None

is_import_text=Text(master=QKS_dataset_import)

#output


QKS_title.pack()
root.mainloop()