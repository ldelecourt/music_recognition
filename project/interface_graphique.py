# coding: utf-8

####################### PARAMS #######################

# repository of the database
p = "."

####################### CONSTANTS ####################

chemin_gif = "trio.gif"


####################### IMPORT #######################

import tkinter
import tkinter.messagebox
import os
import malece


####################### FUNCTIONS #######################
def say_hi():
    tkinter.Label(window, text = "Hi").pack()

    
# Pour le gif
ind = -1
def update(delay=200):
    global ind
    ind += 1
    if ind == 8: ind = 0
    photo.configure(format="gif -index " + str(ind))
    window.after(delay, update)


    
####################### MAIN #######################

# Moving to the directory of the project
os.chdir(p)

# Loading database
df = malece.load_db("malece.pikl") 


# Fenetre principale
window = tkinter.Tk()
# to rename the title of the window
window.title("music recognition")
window.geometry("650x280")
# pack is used to show the object in the window
label = tkinter.Label(window, text = "MUSIC RECOGNITION", fg = "white", bg = "purple").pack()


# creating 2 frames TOP and BOTTOM
top_frame = tkinter.Frame(window).pack()
bottom_frame = tkinter.Frame(window).pack(side = "bottom")


# GIF
can = tkinter.Canvas(top_frame, width=500, height=200, bg='white')
can.pack()
photo = tkinter.PhotoImage(file = chemin_gif)
can.create_image(50, 0, anchor='nw', image=photo, tag='photo')
update()


# Entry
sec = tkinter.StringVar()
txt = tkinter.Entry(bottom_frame, width=15, font = ('arial', 14), textvariable=sec)
txt.pack()
sec.set("Seconds to record")
seconds = sec.get()



# Get the number of seconds submitted by the user
def get_seconds():
    # Disable Entry
    #txt.config(state='disabled')
    # get params to spectrogram
    try:
        params = int(sec.get())
    except ValueError:
        tkinter.messagebox.showinfo("Alert Message", "You must enter an integer only (recommandation : 5)")
        return

    else:
        # On lance la fonction pour record et reconnaitre le son
        resultat = malece.start(df, seconds=params)
        # New window when recognition is done
        window2 = tkinter.Toplevel()
        window2.title("music recognition")
        window2.geometry("400x200")
        artiste, titre, album, annee = resultat
        label_artiste1 = tkinter.Label(window2, text = "Artiste:").pack()
        label_artiste2 = tkinter.Label(window2, text = artiste).pack()
        label_titre1 = tkinter.Label(window2, text = "Titre:").pack()
        label_titre2 = tkinter.Label(window2, text = titre).pack()
        label_album1 = tkinter.Label(window2, text = "Année:").pack()
        label_album2 = tkinter.Label(window2, text = annee).pack()



        
# Boutons
btn = tkinter.Button(bottom_frame, text = "START MUSIC RECOGNITION", fg = "blue", command=get_seconds, font = ('arial', 14, 'bold')).pack()


window.mainloop()

