from tkinter import *

from tkinter.filedialog import *
import PIL.Image
import PIL.ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from traitement import *


class render:

    def __init__(self, fenetre):
        self.filename = ""
        # Configurer la fenêtre
        self.fenetre = fenetre 
        self.fenetre.title("Splines lissantes cubiques") 
        # Création de la partie input
        self.input_side = Frame(fenetre, bg="white")
        self.input_side.grid(row=0, column=0, sticky="NSWE")
        # Fichiers
        self.input_file = Label(self.input_side, text="Fichier", bg="white", font="bold 18")
        self.input_file.grid(row=1, column=0, columnspan = 4, sticky="w", pady=(20, 10))
        self.input_file = Label(self.input_side, text="Sélectionner un fichier", bg="white", font="bold")
        self.input_file.grid(row=2, column=0, columnspan = 4, sticky="w")
        self.button_file = Button(self.input_side, text="Importer",command=self.ouvrir_fichier, fg="#F3421C")
        self.button_file.grid(row=3, column=1, columnspan = 2, sticky="W")
        
        # Afficher les noeuds
        self.noeuds = Label(self.input_side, text="Visualiser les données", bg="white", font="bold 18")
        self.noeuds.grid(row=4, column=0, columnspan = 4, sticky="w")
        self.button_noeuds = Button(self.input_side, text="Afficher",command=self.visualiser_noeuds, fg="#F3421C")
        self.button_noeuds.grid(row=5, column=1, columnspan = 2, sticky="W")
        
        # lambda
        
        self.lissage = Label(self.input_side, text="Paramètrage du lissage", bg="white", font="bold 18")
        self.lissage.grid(row=6, column=0, columnspan = 4, sticky="w", pady=(15, 0), padx =(0, 8))
        
        self.text_lambda = Label(self.input_side, text=u"p", bg="white")
        self.text_lambda.grid(row = 7, column = 0, sticky="NWSE")
        
        self.var_lambda = DoubleVar()
        self.input_lambda = Entry(self.input_side, bg = "white", textvariable = self.var_lambda, width="5")
        self.input_lambda.grid(row = 7, column = 1, sticky = "NWSE")
        
        self.inter = Label(self.input_side, text="fixer un intervalle", bg="white", bd = "5")
        self.inter.grid(row=8, column=0, columnspan = 4, sticky="W", padx=5)
        
        
        self.text_lambdamin = Label(self.input_side, text=u"p inf", bg="white")
        self.text_lambdamin.grid(row = 9, column = 0, sticky="NWSE")
        
        self.var_lambdamin = DoubleVar()
        self.input_lambdamin = Entry(self.input_side, bg = "white", textvariable = self.var_lambdamin, width="5")
        self.input_lambdamin.grid(row = 9, column = 1, sticky = "NWSE")
        

        self.text_lambdamax = Label(self.input_side, text=u"p sup", bg="white")
        self.text_lambdamax.grid(row = 10, column = 0, sticky="NWSE")

        self.var_lambdamax = DoubleVar()
        self.input_lambdamax = Entry(self.input_side, bg = "white", textvariable = self.var_lambdamax, width="5")
        self.input_lambdamax.grid(row = 10, column = 1, sticky = "NWSE")
        

        self.button_calcul = Button(self.input_side, text="Exécuter",command=self.calculer_spline, bg="white", fg="#F3421C")
        self.button_calcul.grid(row=11, column=1, columnspan = 1, sticky="W")

        self.res = Label(self.input_side, text="Résultats", bg="white", bd = "5", font="bold 18")
        self.res.grid(row=12, column=0, sticky="W")

        self.button_CV = Button(self.input_side, text="Courbe Cross Validation",command=self.visualiser_CV, bg="white", fg="#F3421C")
        self.button_CV.grid(row=13, column=0, columnspan = 2, sticky="W")

        self.button_spline = Button(self.input_side, text="Spline Lissante",command=self.visualiser_spline, bg="white", fg="#F3421C")
        self.button_spline.grid(row=14, column=0, columnspan = 2, sticky="W")
        
        self.button_rl = Button(self.input_side, text="Droite de régression",command=self.visualiser_rl, bg="white", fg="#F3421C")
        self.button_rl.grid(row=15, column=0, columnspan = 2, sticky="W")
        
        self.button_interpol = Button(self.input_side, text="Fonction Interpolante",command=self.visualiser_interpol, bg="white", fg="#F3421C")
        self.button_interpol.grid(row=16, column=0, columnspan = 2, sticky="W")
        
        # Création de la partie output
        self.dynamic_screen = Frame(self.fenetre, bg="white")
        self.dynamic_screen.grid(row=0, column=1, sticky="NSEW")
        self.fenetre.grid_columnconfigure(1, weight=1)
        self.fenetre.grid_rowconfigure(0, weight=1)
        self.canvas = Canvas(self.dynamic_screen, bg='white', width=800, height=600)
        self.canvas.grid(sticky="NSEW")
        self.logo = PhotoImage(file=r'/home/lila/Documents/GIS4/semestre1/CN/clspline/logo.png')
        self.canvas.create_image(400,300,image=self.logo)
        
    def ouvrir_fichier(self):

        self.filename = askopenfilename(title="Ouvrir une image",filetypes=[('fichiers txt','.txt')])
        noeuds = read_coordinate(self.filename)
        fig = plt.figure()
        a = plt.subplot(1, 1, 1)
        a.scatter(noeuds[:, 0], noeuds[:, 1],color='red')
        plt.title ("Noeuds", fontsize=16)
        fig.savefig('noeuds.png')

    def visualiser_noeuds(self):
        self.logo = PhotoImage(file=r'/home/lila/Documents/GIS4/semestre1/CN/clspline/noeuds.png')
        self.canvas.create_image(400,300,image=self.logo)

    def calculer_spline(self):
        if(self.filename != "" ):
            splines(self.filename,self.var_lambda.get(), self.var_lambdamin.get(), self.var_lambdamax.get())
    
    def visualiser_CV(self):
        self.logo = PhotoImage(file=r'/home/lila/Documents/GIS4/semestre1/CN/clspline/CV.png')
        self.canvas.create_image(400,300,image=self.logo)
    def visualiser_spline(self):
        self.logo = PhotoImage(file=r'/home/lila/Documents/GIS4/semestre1/CN/clspline/spline.png')
        self.canvas.create_image(400,300,image=self.logo)
    def visualiser_rl(self):
        self.logo = PhotoImage(file=r'/home/lila/Documents/GIS4/semestre1/CN/clspline/rl.png')
        self.canvas.create_image(400,300,image=self.logo)
    def visualiser_interpol(self):
        self.logo = PhotoImage(file=r'/home/lila/Documents/GIS4/semestre1/CN/clspline/interpol.png')
        self.canvas.create_image(400,300,image=self.logo)



fenetre = Tk()

start = render(fenetre)
fenetre.mainloop()
fenetre.destroy()