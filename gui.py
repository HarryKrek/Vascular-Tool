import customtkinter as ctk
from tkinter import ttk
from PIL import Image
from pathlib import Path
import yaml
import asyncio
import os

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
        self.save_button = ctk.CTkButton(self.sidebar_frame, text= 'Save Settings', command= self.save_settings)
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
        #Add button for save location, default to current directory
        self.location_button = ctk.CTkButton(self.processing_scroll, text = "Save Location", command=self.set_save_path)
        self.location_button.grid(row = len(saveAndDisplaySettings), column = 0, padx = 10, pady =10, sticky = 'ew')
        #Set Results location to the current folder as default
        self.save_path = Path.cwd()

        #Log box
        self.logBox = ctk.CTkTextbox(self.tab_select.tab("Log"))
        self.logBox.pack(side = 'bottom', fill = 'both', expand = True, padx= 0, pady = 0)

        # Place loading bar frame at the bottom
        self.bottom_frame = ctk.CTkFrame(self, corner_radius=3)
        self.bottom_frame.grid(row=2, column=0, columnspan=2, sticky="ew")  # Spanning across both columns
        self.bottom_frame.grid_columnconfigure(0, weight=1)  # Adjusted configuration
        #Go Button
        self.run_button = ctk.CTkButton(self.bottom_frame, text = "Run", command= self.run_button_callback)
        self.run_button.grid(row=0,column = 2, sticky = 'nsw')
        self.run_button.grid_columnconfigure(2,weight=1)
        #Loading Bar
        self.load_bar = ctk.CTkProgressBar(self.bottom_frame)
        self.load_bar.grid(row = 0, column = 0, sticky = 'nsew')
        self.load_bar.grid_columnconfigure(0,weight=1)

        self.batch_frame = None
        self.status_indicators = None

        self.single_setup()
        
        self.setup_variables()

    def single_setup(self):
        if self.batch_frame is not None:
            if self.status_table is not None:
                del self.status_table
            del self.batch_frame

        #Image Frame
        self.image = None
        self.imageAfter = None
        self.imageFrame = ctk.CTkTabview(self, height = 1000)
        self.imageFrame.grid(row =0, column =1, padx = 20, pady = 20, sticky = 'nsew')
        self.imageFrame.add("Input Image")
        self.imageFrame.add("Output Image")
        self.beforeImage = ctk.CTkLabel(self.imageFrame.tab("Input Image"), text='', image = self.image)
        self.beforeImage.pack(fill = 'both', expand = True)
        self.afterImage = ctk.CTkLabel(self.imageFrame.tab("Output Image"), text='', image = self.imageAfter)
        self.afterImage.pack(fill = 'both', expand = True)

        self.image_select.configure(text = "Choose Image")

    
    def batch_setup(self):
        #Teardown the single image parts:
        if self.imageFrame is not None:
            del self.imageFrame
        
        # #Add Batch Scroll View
        self.batch_frame = ctk.CTkFrame(self, height = 3000)
        self.batch_frame.grid(row =0, column = 1, padx = 20, pady = 20, sticky = 'nsew')
        # titleFont = ctk.CTkFont("Arial", size = 16, weight = 'bold')
        # self.batch_title = ctk.CTkLabel(self.batch_frame, text="Batch Image Processing Status", height = 80, font=titleFont)
        # self.batch_title.pack(anchor = 'n', expand= False, fill = 'x', pady = 10)
        # self.batch_scroll = ctk.CTkScrollableFrame(self.batch_frame)
        # self.batch_scroll.pack(anchor = 's', expand = True, fill = 'both')

        columns = ("Name", "Status")

        self.status_table = ttk.Treeview(self.batch_frame, columns = columns, show = 'headings')
        self.status_table.heading("Name", text = "Name")
        self.status_table.heading("Status", text = "Status")
        self.status_table.pack(fill='both', expand= True)

        #Change Image selection text
        self.image_select.configure(text = "Select Image Directory")



    def update_batch(self, path):
        self.batch_full_path = os.listdir(path)
        itemNames = [os.path.basename(item) for item in self.batch_full_path]

        #Add names to status as a dictionary
        self.status_indicators = {}
        for i, name in enumerate(itemNames):
            #Row = i
            values = (name, "❌")
            self.status_table.insert(parent = '', index = 0, values=values)
            
            saveVal = {'row': i, 'name':name, 'status':'❌'}
            self.status_indicators[self.batch_full_path[i]] = saveVal
            


    def mode_selection(self, mode):
        if mode == 'Batch Process':
            self.batch = True
            self.batch_setup()
        else:
            self.batch = False
            self.single_setup()

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
        #Single Image Mode
        if self.mode_select == "Single Image":
            self.imgPath = ctk.filedialog.askopenfilenames(initialdir="./", title="Select Settings File", filetypes=(
                ("Image Files", '*.jpg;*.png;*.gif;*.bmp;*.tif;*.tiff'), ("All Files", '*')))[0]

            #Update Image
            self.image = ctk.CTkImage(light_image=Image.open(self.imgPath), dark_image=Image.open(self.imgPath), size=(800,800))
            self.beforeImage.configure(image = self.image)
        
        #Batch Image Mode
        else:
            imgdir = ctk.filedialog.askdirectory(initialdir="./", title = "Select Directory")
            self.update_batch(imgdir)



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
            
            if self.settings_path == "":
                pass
            
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
        #create dictionary from settings already available
        self.config_from_box()
        #Open a file dialog to specify the save location
        file_path = ctk.filedialog.asksaveasfilename(
            defaultextension=".yaml", 
            filetypes=[("YAML files", "*.yaml")], 
            title="Save YAML file"
        )
        
        if file_path == "":
            pass


        # If the user provides a file path, save the dictionary as YAML
        try:
            if file_path:
                with open(file_path, 'w') as yaml_file:
                    yaml.dump(self.config, yaml_file, default_flow_style=False, sort_keys=False)
        except Exception as e:
            FailurePopup(self, str(e))


    def set_save_path(self):
        try:
            self.save_path = ctk.filedialog.askdirectory()
            if self.save_path == "":
                pass
        
        except Exception as e:
            FailurePopup(self, str(e))

    def add_to_log(self, msg):
        self.logBox.insert("0.0", msg + "\n" * 50)

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
                FailurePopup(self, "No Image Loaded")
            if self.config == None:
                FailurePopup(self, "No Config Loaded")
            #TODO run as thread
            result = asyncio.run(run_img(self.imgPath, self.save_path, self.config, "result", 0))
            #Get the resultant image to display
            place = os.path.abspath(str(self.save_path) + "\\" +'result' + ".tif")
            self.imageTwo = ctk.CTkImage(light_image=Image.open(place), dark_image=Image.open(place), size=(800,800))
            self.afterImage.configure(image = self.imageTwo)
            self.add_to_log(str(result))

        except Exception as e:
            FailurePopup(self, str(e))




if __name__ == '__main__':
    app = App()
    app.mainloop()
