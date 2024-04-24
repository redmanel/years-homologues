import tkinter
import tkintermapview
import tkinter.messagebox
import customtkinter
from tkinter import *
import csv
from PIL import Image
import pandas as pd
import os
import datetime
from tkcalendar import Calendar
from pathlib import Path
import main
import homofind_cluster


customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
station_dict = {'Амдерма':'Amderma', 'Голомянный': 'Golomyannyy',
                'Марресаля': 'Marresalya', 'Мыс Болванский Нос': 'Mys_Bolvanskiy_Nos',
                'Мыс Челюскин': 'Mys_Chelyuskin', 'Мыс Сопочная Карга': 'Mys_Sopochnaya_Karga',
                'Мыс Стерлигова': 'Mys_Sterligova', 'Мыс Желания': 'Mys_Zhelaniya',
                'Остров Белый': 'Ostrov_Bely', 'Остров Диксон': 'Ostrov_Dikson',
                'Остров Хейса': 'Ostrov_Kheysa', 'Остров Тройной': 'Ostrov_Troynoy',
                'Остров Визе': 'Ostrov_Vize', 'Сеяха': 'Seyakha', }

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        # configure window
        self.title("weather forecasting by year-homologue")
        self.geometry(f"{1100}x{600}")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.logo_label = customtkinter.CTkLabel(self, text="Homologues", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=0, pady=(0, 0))

        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=1050)
        self.tabview.grid(row=1, column=0, padx=(20, 20), pady=(0, 20), sticky="nsew")
        self.tabview.add("Select a station")
        self.tabview.add("Import")
        self.tabview.add("Search homologues")
        self.tabview.add("Select date")
        self.tabview.add("Forecast plots")
        self.tabview.add("Download")
        self.tabview.tab("Select a station").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Select a station").grid_rowconfigure(0, weight=1)
        self.tabview.tab("Import").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Search homologues").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Select date").grid_columnconfigure(1, weight=1)
        self.tabview.tab("Forecast plots").grid_columnconfigure(1, weight=1)
        self.tabview.tab("Download").grid_columnconfigure(0, weight=1)


        # Select a station tab
        self.map_widget = tkintermapview.TkinterMapView(self.tabview.tab("Select a station"), corner_radius=0)
        self.map_widget.grid(row=0, column=0, rowspan=2, pady=0, padx=0, sticky="nsew")
        self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=m&hl=ru&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)
        self.map_widget.set_position(76.824153, 76.211605)
        self.map_widget.set_zoom(4)
        marker_1 = self.map_widget.set_marker(79.491, 76.981, text="Остров Визе", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_2 = self.map_widget.set_marker(79.550, 90.567, text="Голомянный", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_3 = self.map_widget.set_marker(75.417, 88.900, text="Мыс Стерлигова", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_4 = self.map_widget.set_marker(75.952, 82.946, text="Остров Тройной", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_5 = self.map_widget.set_marker(73.500, 80.400, text="Остров Диксон", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_6 = self.map_widget.set_marker(73.333, 70.050, text="Остров Белый", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_7 = self.map_widget.set_marker(76.950, 68.550, text="Мыс Желания", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_8 = self.map_widget.set_marker(69.717, 66.800, text="Марресаля", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_9 = self.map_widget.set_marker(70.447, 59.091, text="Мыс Болванский Нос", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_10 = self.map_widget.set_marker(70.170, 72.514, text="Сеяха", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_11 = self.map_widget.set_marker(77.717, 104.300, text="Мыс Челюскин", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_12 = self.map_widget.set_marker(71.875, 82.706, text="Мыс Сопочная Карга", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_13 = self.map_widget.set_marker(80.617, 58.050, text="Остров Хейса", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        marker_14 = self.map_widget.set_marker(69.783, 61.783, text="Амдерма", text_color="black",
                                              marker_color_circle="black", marker_color_outside="gray40",
                                              font=("Helvetica Bold", 8), command=self.click_marker)
        self.station_title = customtkinter.CTkLabel(self.tabview.tab("Select a station"),
                                                       text="Selected station",
                                                       font=customtkinter.CTkFont(size=14, weight="bold"))
        self.station_title.grid(row=0, column=1, columnspan=1, padx=(10, 10), pady=(5, 0), sticky='s')
        self.station_entry = customtkinter.CTkEntry(self.tabview.tab("Select a station"),
                                                       placeholder_text="station")
        self.station_entry.grid(row=1, column=1, columnspan=1, padx=(10, 10), pady=(5, 0), sticky="s")


        # Import tab
        self.file_name = customtkinter.CTkLabel(self.tabview.tab("Import"),
                                                text=f'Required data structure: date, maximum temperature,'
                                                     f' minimum temperature, average temperature, wind speed,'
                                                     f' precipitation, effective temperature',
                                                font=customtkinter.CTkFont(size=14, weight="bold"))
        self.file_name.grid(row=0, column=0, padx=(20, 20), pady=(10, 10))
        self.import_buttom = customtkinter.CTkButton(self.tabview.tab("Import"), text="Select file", width=1000,
                                                     font=customtkinter.CTkFont(size=16, weight="bold"),
                                                     command=self.import_buttom_event)
        self.import_buttom.grid(row=1, column=0, padx=(20, 20), pady=(10, 10), sticky="ew")


        # Search homologues tab
        self.search_homologues_buttom = customtkinter.CTkButton(self.tabview.tab("Search homologues"),
                                                                text="Search homologues", width=1000,
                                                     font=customtkinter.CTkFont(size=16, weight="bold"),
                                                     command=self.search_homologues_buttom_event)
        self.search_homologues_buttom.grid(row=0, column=0, padx=(20, 20), pady=(10, 10), sticky="ew")


        # Select Date tab
        # start date title
        self.start_date_title = customtkinter.CTkLabel(self.tabview.tab("Select date"),
                                                       text="Start Date",
                                                       font=customtkinter.CTkFont(size=14, weight="bold"))
        self.start_date_title.grid(row=0, column=0, columnspan=1, padx=(140, 20), pady=10, sticky='w')
        # start date calander
        self.start_cal = Calendar(self.tabview.tab("Select date"), selectmode='day',
                                  showweeknumbers=False, cursor="hand2", date_pattern='y-mm-dd',
                                  borderwidth=0, bordercolor='white')
        self.start_cal.grid(row=1, column=0, padx=(100, 20), pady=10, sticky='w')

        # end date title
        self.end_date_title = customtkinter.CTkLabel(self.tabview.tab("Select date"),
                                                     text="End Date", font=customtkinter.CTkFont(size=14, weight="bold"))
        self.end_date_title.grid(row=0, column=1, columnspan=1, padx=(20, 140), pady=10, sticky='e')
        # end date calander
        self.end_cal = Calendar(self.tabview.tab("Select date"), selectmode='day',
                                showweeknumbers=False, cursor="hand2", date_pattern='y-mm-dd',
                                borderwidth=0, bordercolor='white')
        self.end_cal.grid(row=1, column=1, padx=(20, 100), pady=10, sticky='e')
        # start date entry
        self.start_date_entry = customtkinter.CTkEntry(self.tabview.tab("Select date"),
                                                       placeholder_text="Start Date")
        self.start_date_entry.grid(row=3, column=0, padx=(100, 20), pady=10, sticky="w")
        # end date entry
        self.end_date_entry = customtkinter.CTkEntry(self.tabview.tab("Select date"),
                                                     placeholder_text="End Date")
        self.end_date_entry.grid(row=3, column=1, padx=(20, 100), pady=10, sticky="e")
        # date confirm button
        self.confirm_date_buttom = customtkinter.CTkButton(self.tabview.tab("Select date"),
                                                    text="Confirm Dates", hover=True,
                                                    font=customtkinter.CTkFont(size=16, weight="bold"),
                                                    command=self.fetch_dates_overall)
        self.confirm_date_buttom.grid(row=2, column=0, columnspan=2, padx=10, pady=30)


        # for Forecast tab
        self.forecast_buttom = customtkinter.CTkButton(self.tabview.tab("Select date"), text="Predict values", width=200,
                                                     font=customtkinter.CTkFont(size=16, weight="bold"),
                                                     command=self.forecast_buttom_event)
        self.forecast_buttom.grid(row=4, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="nsew")
        self.optionmenu_plots = customtkinter.CTkOptionMenu(self.tabview.tab("Select date"), dynamic_resizing=False,
                                                        values=["maximum temperature", "minimum temperature", "average temperature",
                                                                "wind speed", "precipitation", "effective temperature"],
                                                            width=200)
        self.optionmenu_plots.grid(row=5, column=0, padx=20, pady=(10, 10), sticky="nsew")
        self.plots_buttom = customtkinter.CTkButton(self.tabview.tab("Select date"), text="Show forecast plot", width=200,
                                                     font=customtkinter.CTkFont(size=16, weight="bold"),
                                                     command=self.plots_buttom_event)
        self.plots_buttom.grid(row=5, column=1,  padx=20, pady=(10, 10), sticky="nsew")


        # Download tab
        self.show_forecast_data_buttom = customtkinter.CTkButton(self.tabview.tab("Download"), text="Show forecast data", width=470,
                                                     font=customtkinter.CTkFont(size=16, weight="bold"),
                                                     command=self.show_forecast_data_event)
        self.show_forecast_data_buttom.grid(row=0, column=0, padx=20, pady=(10, 10), sticky="ew")
        self.download_buttom = customtkinter.CTkButton(self.tabview.tab("Download"), text="Download", width=470,
                                                     font=customtkinter.CTkFont(size=16, weight="bold"),
                                                     command=self.download_buttom_event)
        self.download_buttom.grid(row=0, column=1, padx=20, pady=(10, 10), sticky="ew")


    # buttoms

    # choose station
    def click_marker(self, marker):
        station = marker.text
        self.station_entry.configure(placeholder_text=station)
        if station in station_dict.keys():
            name = station_dict.get(station)
            path = str(Path(__file__).parent) + f'/stations/{name}_after_preprocessing.csv'
        global abs_path
        abs_path = os.path.abspath(path)
        global station_name
        station_name = name
        print("marker clicked")


    # import file
    def import_buttom_event(self):
        global filename
        filename = customtkinter.filedialog.askopenfilename()
        # first date value
        f_read = open(filename, "r")
        last_date = f_read.readlines()[-1].split(',')[0]
        global first_date
        first_date = datetime.datetime.strptime(last_date, '%Y-%m-%d').date() + datetime.timedelta(days=1)
        self.start_cal.selection_set(first_date)
        self.start_cal.configure(state='disabled', disabledselectbackground='steelblue')
        self.end_cal.selection_set(first_date)
        # Table style
        style = tkinter.ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background="#2a2d2e",
                        foreground="white",
                        rowheight=25,
                        fieldbackground="#343638",
                        bordercolor="#343638",
                        borderwidth=0)
        style.map('Treeview', background=[('selected', '#22559b')])

        style.configure("Treeview.Heading",
                        background="#565b5e",
                        foreground="white",
                        relief="flat")
        style.map("Treeview.Heading",
                  background=[('active', '#3484F0')])
        # Table
        tree = tkinter.ttk.Treeview(self.tabview.tab("Import"), show="headings", height=100)
        tree.grid(row=3, column=(0), padx=(20, 20), pady=(10, 20))
        status_label = tkinter.Label(self.tabview, text="", padx=20, pady=10)
        with open(filename, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Read the header row
            tree.delete(*tree.get_children())  # Clear the current data

            tree["columns"] = header
            for col in header:
                tree.heading(col, text=col)
                tree.column(col, width=150)

            for row in csv_reader:
                tree.insert("", "end", values=row)
                status_label.config(text=f"CSV file loaded: {filename}")
        print("import_buttom clicked")


    #
    def search_homologues_buttom_event(self):
        global homologues
        homologues, years_dict = main.homofind(abs_path, filename)
        cluster_figs = homofind_cluster.visualize_clusters(years_dict, homologues)
        print(homologues)

        # cluster_figs - list с названиями файлов - графиков кластеров для каждого параметра
        path = str(Path(__file__).parent) + '/' +cluster_figs[0]
        my_image = customtkinter.CTkImage(light_image=Image.open(path),
                                          dark_image=Image.open(path),
                                          size=(600, 450))
        image_label = customtkinter.CTkLabel(self.tabview.tab("Search homologues"), image=my_image, text="")
        image_label.grid(row=1, column=0, padx=(20, 20), pady=(10, 20))


    # select date
    def fetch_dates_overall(self):
        global date1
        date1 = self.start_cal.get_date()
        self.start_date_entry.configure(placeholder_text=date1)
        self.start_date_entry.configure(state='readonly')
        global date2
        date2 = self.end_cal.get_date()
        self.end_date_entry.configure(placeholder_text=date2)
        self.end_date_entry.configure(state='readonly')
        global n_days
        n_days = pd.to_datetime(date2) - pd.to_datetime(date1)
        n_days = n_days.days
        print(n_days)
        print('confirm_date_buttom clicked')


    # forecast plots
    def forecast_buttom_event(self):
        global preds
        global pred_figs
        preds, pred_figs = main.get_predicts(abs_path, filename, n_days, homologues, station_name)
        print("forecast_buttom clicked")

    def plots_buttom_event(self):
        parameter = self.optionmenu_plots.get()
        if parameter == "maximum temperature":
            parameter_col = 0
            label = 'temperature'
            title = "maximum_temperature"
        elif parameter == "minimum temperature":
            parameter_col = 1
            label = 'temperature'
            title = "minimum_temperature"
        elif parameter == "average temperature":
            parameter_col = 1
            label = 'temperature'
            title = "average_temperature"
        elif parameter == "wind speed":
            parameter_col = 3
            label = 'wind speed'
            title = "wind_speed"
        elif parameter == "precipitation":
            parameter_col = 4
            label = 'precipitation'
            title = "precipitation"
        elif parameter == "effective temperature":
            parameter_col = 5
            label = 'temperature'
            title = "effective_temperature"
        path_pred = rf'{pred_figs[title]}'
        path_pred = os.path.abspath(path_pred)
        my_image = customtkinter.CTkImage(light_image=Image.open(path_pred),
                                          dark_image=Image.open(path_pred),
                                          size=(800, 500))
        image_label = customtkinter.CTkLabel(self.tabview.tab("Forecast plots"), image=my_image, text="")
        image_label.grid(row=1, column=0, columnspan=2, padx=(20, 20), pady=(10, 20))
        print("plots_buttom clicked")

    # show_forecast_data
    def show_forecast_data_event(self):
        # Table style
        style = tkinter.ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background="#2a2d2e",
                        foreground="white",
                        rowheight=25,
                        fieldbackground="#343638",
                        bordercolor="#343638",
                        borderwidth=0)
        style.map('Treeview', background=[('selected', '#22559b')])

        style.configure("Treeview.Heading",
                        background="#565b5e",
                        foreground="white",
                        relief="flat")
        style.map("Treeview.Heading",
                  background=[('active', '#3484F0')])
        # Table
        tablefile = main.dict_to_df(preds)
        tablefile.reset_index(inplace= True)
        tablefile = tablefile.rename(columns={'index': 'date'})
        tablefile['date'] = pd.to_datetime(tablefile['date']).dt.date
        columns = tablefile.columns.values.tolist()
        print(tablefile)
        tree = tkinter.ttk.Treeview(self.tabview.tab("Download"), columns=columns, show="headings", height=100)
        tree.grid(row=1, column=0, columnspan=2, padx=(20, 20), pady=(10, 20))
        status_label = tkinter.Label(self.tabview, text="", padx=20, pady=20)
        for col in columns:
            tree.heading(col,text=col)
            tree.column(col, width=100)
        for index, row in tablefile.iterrows():
            tree.insert("", END, values=row.tolist())
            status_label.config(text=f"CSV file loaded: {filename}")

        print("show_forecast_data clicked")

    # download forecast data
    def download_buttom_event(self):
        types = [("CSV files", ".csv")]
        dataFile = main.dict_to_df(preds)
        SAVING_PATH = tkinter.filedialog.asksaveasfile(defaultextension=".csv", filetypes=types)
        dataFile.to_csv(SAVING_PATH, lineterminator='\n')
        print("download_buttom clicked")


if __name__ == "__main__":
    app = App()
    app.mainloop()