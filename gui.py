import customtkinter as ctk
from PIL import Image

class SettingFrame(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master)


class MyScrollableCheckboxFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, title, values):
        super().__init__(master, label_text=title)
        self.grid_columnconfigure(0, weight=1)
        self.values = values
        self.checkboxes = []

        for i, value in enumerate(self.values):
            checkbox = ctk.CTkCheckBox(self, text=value)
            checkbox.grid(row=i, column=0, padx=10, pady=(10, 0), sticky="w")
            self.checkboxes.append(checkbox)

    def get(self):
        checked_checkboxes = []
        for checkbox in self.checkboxes:
            if checkbox.get() == 1:
                checked_checkboxes.append(checkbox.cget("text"))
        return checked_checkboxes


class App(ctk.CTk):
    def __init__(self):
        # Initiate super
        super().__init__()
            
        # Create Grid
        self.title("BRAT -  Vascular Image Analysis")
        self.geometry(f"{1100}x{580}")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)  # Added a new row for the bottom frame
        
        # Create sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=3)
        self.sidebar_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(3, weight=1)

        # Place Items in Sidebar Frame
        # Firstly BRAT Frame
        self.logo = ctk.CTkLabel(self.sidebar_frame, width=100, height=60, corner_radius=5, text="BRAT", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo.grid(row=0, column=0, sticky="new", padx=10, pady=10)

        #Frame with label and dropdown menu to select single vs multi mode
        self.selection_frame = ctk.CTkFrame(self.sidebar_frame)
        self.selection_frame.grid(row = 1, column = 0, padx = 10, pady= 10, sticky = 'new')
        self.selection_frame.grid_rowconfigure(1, weight = 1)
        #Fill this with selection
        #MODE Label
        temp =ctk.CTkLabel(self.selection_frame, text = "Selection Mode", font=ctk.CTkFont(size=20, weight="bold"))
        temp.grid(row=0, column = 0, columnspan = 1, sticky = 'ew')
        #
        self.mode_select = ctk.CTkOptionMenu(self.selection_frame, dynamic_resizing = False,
                                             values = ["Single Image", "Batch Process"])
        self.mode_select.grid(row = 0, column= 1, columnspan =1, padx =10, pady=10, sticky = 'ew')

        # Place loading bar frame at the bottom
        self.bottom_frame = ctk.CTkFrame(self, corner_radius=3)
        self.bottom_frame.grid(row=2, column=0, columnspan=2, sticky="ew")  # Spanning across both columns
        self.bottom_frame.grid_columnconfigure(0, weight=1)  # Adjusted configuration
        #Go Button
        self.run_button = ctk.CTkButton(self.bottom_frame)
        self.run_button.grid(row=0,column = 2, sticky = 'nsw')
        self.run_button.grid_columnconfigure(2,weight=1)
        #Loading Bar
        self.load_bar = ctk.CTkProgressBar(self.bottom_frame)
        self.load_bar.grid(row = 0, column = 0, sticky = 'nsew')
        self.load_bar.grid_columnconfigure(0,weight=1)


        #Image Frame
        #TODO AUTOSCALING FOR IMAGE
        self.image = ctk.CTkImage(light_image=Image.open('WT_1.tif'), dark_image=Image.open('WT_1.tif'), size=(1200,1200))
        self.imageFrame = ctk.CTkLabel(self, text='', image = self.image)
        self.imageFrame.grid(row =0, column =1, padx = 20, pady = 20, sticky = 'nsew')


        


if __name__ == '__main__':
    app = App()
    app.mainloop()
