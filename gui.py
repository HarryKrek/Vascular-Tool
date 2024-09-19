import customtkinter as ctk
from PIL import Image
from pathlib import Path

#Import components from vascular tool
# from vascular_tool import

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
        self.geometry(f"{1500}x{680}")
        self.minsize(1500, 680)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)  # Added a new row for the bottom frame

        # Create sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self, width=200, height= 1000, corner_radius=3)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, columnspan = 1, sticky="nsew")
        self.sidebar_frame.grid_columnconfigure(0, weight =1)
        self.sidebar_frame.grid_rowconfigure(2, weight = 5 )

        # Place Items in Sidebar Frame
        # Buttons for selection of settings and images
        self.image_select = ctk.CTkButton(self.sidebar_frame, text = "Choose Image", corner_radius = 3, height = 100, command = self.image_callback)
        self.image_select.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'ew')
        #Image selection button
        self.settings_select = ctk.CTkButton(self.sidebar_frame, text = "Settings File", corner_radius=3, height = 100, command = self.settings_callback)
        self.settings_select.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'ew')


        #Fill this with selection
        #MODE Label
        self.mode_select = ctk.CTkOptionMenu(self.sidebar_frame, dynamic_resizing = False,
                                             values = ["Single Image", "Batch Process"], 
                                             command = self.mode_select())
        self.mode_select.grid(row = 1, column= 0, columnspan =2, padx =10, pady=10, sticky = 'new')

        #Save/Load Settings
        self.save_button = ctk.CTkButton(self.sidebar_frame, text= 'Save Settings')
        self.save_button.grid(row = 2, column = 0, sticky = 'new', padx = 10, pady = 10, columnspan = 2)
        self.load_button = ctk.CTkButton(self.sidebar_frame, text = 'Load Settings')
        self.load_button.grid(row = 3, column = 0, sticky = 'new', padx = 10, pady = 10, columnspan = 2)

        #Add tab selections
        self.tab_select = ctk.CTkTabview(self.sidebar_frame, height = 400)
        self.tab_select.grid(row=4, column = 0, sticky = 'nsew', columnspan = 2, rowspan = 2)
        self.tab_select.add("Analysis")
        self.tab_select.add("Processing")
        self.tab_select.add("Log")
        self.tab_select.set("Analysis")
        self.tab_select.grid_columnconfigure(0, weight=1)
        self.tab_select.grid_rowconfigure(0, weight=1)

        #Add settings to the analysis tab
        self.analysis_scroll = ctk.CTkScrollableFrame(self.tab_select.tab("Analysis"))
        self.analysis_scroll.pack(side ='bottom', fill = 'both', expand = 'true', padx = 0, pady= 0)
        #Setup the rows and columns for this
        self.analysis_scroll.grid_columnconfigure(0, weight = 2)
        self.analysis_scroll.grid_columnconfigure(1, weight= 1)
        self.analysis_scroll.grid_rowconfigure(0, weight=1)

        # Add Settings to the
        analysisSettings = ['Blur Sigma', 'Min Hole Size', 'Min Object Size', 'Min Spur Line Length',
                            'Min Length for internal Line', 'Min Vessel Width']
        self.entries = []
        for i, setting in enumerate(analysisSettings):
            textTemp = ctk.CTkLabel(self.analysis_scroll, text = setting)
            textTemp.grid(row = i, column = 0, padx = 10, pady = 10, sticky = 'ew')
            entry = ctk.CTkEntry(self.analysis_scroll)
            entry.grid(row = i, column = 1, padx = 10, pady = 10, sticky = 'ew')

        # Place loading bar frame at the bottom
        self.bottom_frame = ctk.CTkFrame(self, corner_radius=3)
        self.bottom_frame.grid(row=2, column=0, columnspan=2, sticky="ew")  # Spanning across both columns
        self.bottom_frame.grid_columnconfigure(0, weight=1)  # Adjusted configuration
        #Go Button
        self.run_button = ctk.CTkButton(self.bottom_frame, text = "Run")
        self.run_button.grid(row=0,column = 2, sticky = 'nsw')
        self.run_button.grid_columnconfigure(2,weight=1)
        #Loading Bar
        self.load_bar = ctk.CTkProgressBar(self.bottom_frame)
        self.load_bar.grid(row = 0, column = 0, sticky = 'nsew')
        self.load_bar.grid_columnconfigure(0,weight=1)


        #Image Frame
        self.image = None
        self.imageFrame = ctk.CTkLabel(self, text='', image = self.image)
        self.imageFrame.grid(row =0, column =1, padx = 20, pady = 20, sticky = 'nsew')

        self.setup_variables()

    def mode_select(self, mode):
        #TODO SETUP CHANGE IN VIEW WHEN DOING BATCH SETUP
        if mode == 'Batch Process':
            self.batch = True
        else:
            self.batch = False

    def setup_variables(self):
        self.imgPath = None
        self.settingsPath = None
        self.batch = False


    def settings_callback(self):
        self.settingsPath = ctk.filedialog.askopenfilenames(initialdir="./", title="Select Settings File", filetypes=(("YAML Files",
                                                        "*.y?ml*"),
                                                       ("All Files",
                                                        "*.*")))

    def image_callback(self):
        # if self.mode_selec
        self.imgPath = ctk.filedialog.askopenfilenames(initialdir="./", title="Select Settings File", filetypes=(
            ("Image Files", '*.jpg;*.png;*.gif;*.bmp;*.tif;*.tiff'), ("All Files", '*')))[0]

        #Update Image
        self.image = ctk.CTkImage(light_image=Image.open(self.imgPath), dark_image=Image.open('WT_1.tif'), size=(800,800))
        self.imageFrame.configure(image = self.image)


    def load_settings(self):
        pass

    def save_settings(self):
        pass

    def run_tool_single(self):
        pass




if __name__ == '__main__':
    app = App()
    app.mainloop()
