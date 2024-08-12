import customtkinter as ctk


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
        self.logo = ctk.CTkLabel(self, width=120, height=120, corner_radius=1, text="BRAT", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo.grid(row=0, column=0, sticky="new", padx=10, pady=10)

        # Place loading bar frame at the bottom
        self.bottom_frame = ctk.CTkFrame(self, corner_radius=3)
        self.bottom_frame.grid(row=2, column=0, columnspan=2, sticky="ew")  # Spanning across both columns
        self.bottom_frame.grid_columnconfigure(0, weight=1)  # Adjusted configuration



if __name__ == '__main__':
    app = App()
    app.mainloop()
