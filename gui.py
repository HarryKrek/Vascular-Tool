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
        self.sidebar_frame = ctk.CTkFrame(self, width=200, height= 1000, corner_radius=3)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, columnspan = 1, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(0, weight=1)
        self.sidebar_frame.grid_columnconfigure(0, weight =1)
        self.sidebar_frame.grid_rowconfigure(2, weight =3 )

        # Place Items in Sidebar Frame
        # Buttons for selection of settings and images
        self.image_select = ctk.CTkButton(self.sidebar_frame, text = "Choose Image", corner_radius = 3)
        self.image_select.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'nsew')
        #Image selection button
        self.settings_select = ctk.CTkButton(self.sidebar_frame, text = "Settings File", corner_radius=3)
        self.settings_select.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'nsew')

        
        #Fill this with selection
        #MODE Label
        self.modeLabel =ctk.CTkLabel(self.sidebar_frame, text = "Selection Mode", font=ctk.CTkFont(size=20, weight="bold"))
        self.modeLabel.grid(row=1, column = 0, columnspan = 1, sticky = 'ew')
        #
        self.mode_select = ctk.CTkOptionMenu(self.sidebar_frame, dynamic_resizing = False,
                                             values = ["Single Image", "Batch Process"])
        self.mode_select.grid(row = 1, column= 1, columnspan =1, padx =10, pady=10, sticky = 'new')


        #Add tab selections
        self.tab_select = ctk.CTkTabview(self.sidebar_frame)
        self.tab_select.grid(row=2, column =0, sticky = 'nsew', columnspan = 3, rowspan = 2)
        self.tab_select.add("Analysis")
        self.tab_select.add("Processing")
        self.tab_select.set("Analysis")
        self.tab_select.grid_columnconfigure(0, weight=1)
        self.tab_select.grid_rowconfigure(0, weight=1)

        #Add settings to the analysis tab
        self.analysis_scroll = ctk.CTkScrollableFrame(self.tab_select.tab("Analysis"))
        self.analysis_scroll.pack(fill = 'both')

        

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
