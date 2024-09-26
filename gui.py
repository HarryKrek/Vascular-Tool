import customtkinter as ctk
from PIL import Image
from pathlib import Path
import yaml

analysisSettings = ['Blur Sigma', 'Min Hole Size', 'Min Object Size', 'Min Spur Line Length',
                            'Min Length for Internal Line', 'Minimum Vessel Width']
saveAndDisplaySettings = ['Save Image', 'Show Image']


#Import components from vascular tool
from vascular_tool import run_img, run_batch

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


class FailurePopup(ctk.CTkToplevel):
    def __init__(self, parent, message):
        super().__init__(parent)
        self.title("Exception")
        self.geometry("300x150")

        # Create a label to display the message
        lbl_message = ctk.CTkLabel(self, text=message)
        lbl_message.pack(padx=20, pady=20)

        # Create a dismiss button
        btn_dismiss = ctk.CTkButton(self, text="Dismiss", command=self.destroy)
        btn_dismiss.pack()

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
                                             command = self.mode_selection)
        self.mode_select.grid(row = 1, column= 0, columnspan =2, padx =10, pady=10, sticky = 'new')

        #Save/Load Settings
        self.save_button = ctk.CTkButton(self.sidebar_frame, text= 'Save Settings')
        self.save_button.grid(row = 2, column = 0, sticky = 'new', padx = 10, pady = 10, columnspan = 2)
        self.load_button = ctk.CTkButton(self.sidebar_frame, text = 'Load Settings', command=self.load_settings)
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

        # Add Settings to the grid
        self.entries = {}
        for i, setting in enumerate(analysisSettings):
            textTemp = ctk.CTkLabel(self.analysis_scroll, text = setting)
            textTemp.grid(row = i, column = 0, padx = 10, pady = 10, sticky = 'ew')
            entry = ctk.CTkEntry(self.analysis_scroll)
            entry.grid(row = i, column = 1, padx = 10, pady = 10, sticky = 'ew')
            self.entries[setting] = entry

        #Setup processing area
        self.processing_scroll = ctk.CTkScrollableFrame(self.tab_select.tab("Processing"))
        self.processing_scroll.pack(side = 'bottom', fill = 'both', expand = True, padx =0, pady = 0)
        self.processing_scroll.grid_columnconfigure(0, weight = 2)
        self.processing_scroll.grid_columnconfigure(1, weight= 1)
        self.processing_scroll.grid_rowconfigure(0, weight=1)
        
        #Add processing settings to grid
        for i, setting in enumerate(saveAndDisplaySettings):
            textTemp = ctk.CTkLabel(self.processing_scroll, text = setting)
            textTemp.grid(row = i, column = 0, padx = 10, pady = 10, sticky = 'ew')
            entry = ctk.CTkCheckBox(self.processing_scroll, text="")
            entry.grid(row = i, column = 1, padx = 10, pady = 10, sticky = 'ew')
            self.entries[setting] = entry

        #Log box
        self.logBox = ctk.CTkTextbox(self.tab_select.tab("Log"))
        self.logBox.pack(side = 'bottom', fill = 'both', expand = True, padx= 0, pady = 0)

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

    def mode_selection(self, mode):
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



    def config_from_box(self):
        self.config = {}
        #Extract
        for key in (analysisSettings + saveAndDisplaySettings):
            self.config[key] = self.entries[key].get()
        

    def load_settings(self):
        #Load Settings
        try:
            #Clear config and settings inputs
            self.config = {}
            for key in (analysisSettings + saveAndDisplaySettings):
                if type(self.entries[key]) == ctk.CTkEntry:
                    self.entries[key].delete(0, -1)

            self.settings_path = Path(ctk.filedialog.askopenfilenames(initialdir="./", title="Select Settings File", filetypes=(
                ("yaml", '*.yml;*.yaml'), ("All Files", '*')))[0])
        
            #Load settings from yaml file
            file = open(self.settings_path, 'r')
            config_loaded = yaml.safe_load(file)
            #Load in the settings from the yaml

            for key in (analysisSettings + saveAndDisplaySettings):
                #Add to new config 
                self.config[key] = config_loaded[key]
                if type(self.entries[key]) == ctk.CTkEntry:
                    #Set entry
                    self.entries[key].insert(0, str(config_loaded[key]))
                elif type(self.entries[key] == ctk.CTkCheckBox):
                    if config_loaded[key]:
                        self.entries[key].select()
                    else:
                        self.entries[key].deselect()
        except Exception as e:
            #Pop up with exception if failiure
            FailurePopup(self, str(e))
        

    def save_settings(self):
        pass
        #create dictionary from settings already available

        #Save dictionary with popup

    def run_button_callback(self):
        #Pull settings from dialogue boxes

        #Run Relevant process
        if self.batch:
            self.run_tool_batch()
        else:
            self.run_tool_single()

    def run_tool_batch(self):
        pass

    def run_tool_single(self):
        try:
            if self.image == None:
                #Raise Error
                raise ValueError("No Image Loaded")
            if self.config == None:
                raise ValueError("No Config Loaded")

            run_img(self.image, self.resultsPath, self.config, self.save_name, 0)
        except Exception as e:
            FailurePopup(self, str(e))




if __name__ == '__main__':
    app = App()
    app.mainloop()