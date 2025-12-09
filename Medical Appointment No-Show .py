import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], errors='coerce')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')
    df.dropna(subset=['ScheduledDay', 'AppointmentDay'], inplace=True)
    df['lead_time'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df = df[df['Age'] >= 0]
    df = df[df['lead_time'] >= 0]
    return df

def train_model(df):
    features = [
        'Age', 'Scholarship', 'Hipertension', 'Diabetes',
        'Alcoholism', 'SMS_received', 'lead_time', 'Handcap', 'Gender'
    ]

    X = df[features]
    y = df['No-show']
    X = pd.get_dummies(X, columns=['Gender'], drop_first=True, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced_subsample',
        random_state=42
    )

    model.fit(X_train, y_train)
    print("FINAL MODEL ACCURACY:", model.score(X_test, y_test))
    return model, X_train.columns

def predict_new_patient(model, input_data, columns):
    df = pd.DataFrame(input_data)
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True, dtype=int)
    df = df.reindex(columns=columns, fill_value=0)
    pred = model.predict(df)
    prob = model.predict_proba(df)
    return pred[0], prob[0][1]

def display_random_data(df):
    return df.sample(n=20)

class NoShowPredictorApp:
    def __init__(self, master, df):
        self.master = master
        master.title("No-Show Prediction")
        self.master.attributes('-fullscreen', True)
        self.master.bind("<Escape>", self.exit_fullscreen)
        self.is_dark = False

        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.title_label = tk.Label(self.main_frame, text="Medical Appointment No-Show Predictor",
                                    font=('Helvetica', 18, 'bold'))
        self.title_label.pack(pady=15)

        self.input_frame = tk.LabelFrame(self.main_frame, text="Patient Information",
                                         padx=20, pady=15, font=('Helvetica', 12))
        self.input_frame.pack(fill=tk.X, padx=50, pady=10)

        # ------------- INPUT FIELDS ----------------
        labels = [
            "Age", "Scholarship (0/1)", "Hypertension (0/1)",
            "Diabetes (0/1)", "Alcoholism (0/1)", "SMS Received (0/1)",
            "Lead Time (days)", "Gender (M/F)"
        ]

        self.entries = {}
        self.all_labels = []

        for i, label_text in enumerate(labels):
            row = i // 3
            col = (i % 3) * 2
            label = tk.Label(self.input_frame, text=label_text, font=('Helvetica', 11))
            label.grid(row=row, column=col, sticky="e", pady=8, padx=5)
            self.all_labels.append(label)

            entry = tk.Entry(self.input_frame, width=15, font=('Helvetica', 11))
            entry.grid(row=row, column=col+1, pady=8, padx=5)
            self.entries[label_text] = entry

        # Handicap dropdown
        label = tk.Label(self.input_frame, text="Handicap (0–4)", font=('Helvetica', 11))
        label.grid(row=3, column=0, sticky="e", pady=8, padx=5)
        self.all_labels.append(label)

        self.handcap_var = tk.StringVar()
        self.handcap_dropdown = ttk.Combobox(
            self.input_frame, textvariable=self.handcap_var, values=["0","1","2","3","4"],
            state="readonly", font=('Helvetica', 11), width=13
        )
        self.handcap_dropdown.grid(row=3, column=1, pady=8, padx=5)
        self.handcap_dropdown.current(0)

        # ---------------- BUTTONS ---------------------
        button_frame = tk.Frame(self.main_frame)
        button_frame.pack(pady=15)

        self.predict_button = tk.Button(button_frame, text="Predict", command=self.predict,
                                        bg="#4CAF50", fg="white",
                                        width=20, font=('Helvetica', 12, 'bold'))
        self.predict_button.grid(row=0, column=0, padx=20)

        self.show_data_button = tk.Button(button_frame, text="Show Random Data",
                                          command=self.show_random_data,
                                          bg="#2196F3", fg="white",
                                          width=20, font=('Helvetica', 12, 'bold'))
        self.show_data_button.grid(row=0, column=1, padx=20)

        self.exit_button = tk.Button(button_frame, text="Exit Full Screen",
                                     command=self.exit_fullscreen,
                                     bg="#f44336", fg="white",
                                     width=20, font=('Helvetica', 12, 'bold'))
        self.exit_button.grid(row=0, column=2, padx=20)

        self.theme_button = tk.Button(button_frame, text="Dark Mode",
                                      command=self.toggle_theme,
                                      bg="#555555", fg="white",
                                      width=20, font=('Helvetica', 12, 'bold'))
        self.theme_button.grid(row=0, column=3, padx=20)

        # ------------------- RESULT ------------------------
        result_frame = tk.LabelFrame(self.main_frame, text="Prediction Result", font=('Helvetica', 12))
        result_frame.pack(fill=tk.X, padx=50, pady=15)

        self.result_label = tk.Label(result_frame, text="No prediction yet",
                                     font=('Helvetica', 16, 'bold'), pady=15)
        self.result_label.pack()

        # ------------------- PIE CHART AREA ------------------------
        self.pie_frame = tk.LabelFrame(self.main_frame, text="Patient Outcome Probability",
                                       font=('Helvetica', 12))
        self.pie_frame.pack(fill=tk.X, padx=50, pady=15)

        self.pie_canvas = None

        # ------------------- RANDOM DATA TABLE ------------------------
        data_frame = tk.LabelFrame(self.main_frame, text="Sample Data", font=('Helvetica', 12))
        data_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=10)

        self.data_text = tk.Text(data_frame, height=10, font=('Courier', 10), wrap=tk.NONE)
        self.data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrolly = tk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.data_text.yview)
        scrolly.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_text['yscrollcommand'] = scrolly.set

        scrollx = tk.Scrollbar(data_frame, orient=tk.HORIZONTAL, command=self.data_text.xview)
        scrollx.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_text['xscrollcommand'] = scrollx.set

        self.df = df

    # -----------------------------------------------------

    def toggle_theme(self):
        if not self.is_dark:
            bg = "#121212"; fg = "white"; entry_bg = "#1e1e1e"; frame_bg = "#181818"
            self.theme_button.config(text="Light Mode")
        else:
            bg = "white"; fg = "black"; entry_bg = "white"; frame_bg = "white"
            self.theme_button.config(text="Dark Mode")

        self.master.config(bg=bg)
        self.main_frame.config(bg=bg)
        self.title_label.config(bg=bg, fg=fg)
        self.input_frame.config(bg=frame_bg, fg=fg)
        self.pie_frame.config(bg=frame_bg, fg=fg)
        self.result_label.config(bg=frame_bg, fg=fg)
        self.data_text.config(bg=entry_bg, fg=fg)

        for label in self.all_labels:
            label.config(bg=frame_bg, fg=fg)

        for entry in self.entries.values():
            entry.config(bg=entry_bg, fg=fg, insertbackground=fg)

        self.handcap_dropdown.config(background=entry_bg, foreground=fg)

        self.predict_button.config(bg="#2e7d32" if not self.is_dark else "#4CAF50")
        self.show_data_button.config(bg="#0d47a1" if not self.is_dark else "#2196F3")
        self.exit_button.config(bg="#b71c1c" if not self.is_dark else "#f44336")

        self.is_dark = not self.is_dark

    # -----------------------------------------------------

    def show_pie_chart(self, values, labels):
        if self.pie_canvas:
            self.pie_canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')

        self.pie_canvas = FigureCanvasTkAgg(fig, master=self.pie_frame)
        self.pie_canvas.draw()
        self.pie_canvas.get_tk_widget().pack()

    # -----------------------------------------------------

    def predict(self):
        try:
            input_data = {
                'Age': [int(self.entries["Age"].get())],
                'Scholarship': [int(self.entries["Scholarship (0/1)"].get())],
                'Hipertension': [int(self.entries["Hypertension (0/1)"].get())],
                'Diabetes': [int(self.entries["Diabetes (0/1)"].get())],
                'Alcoholism': [int(self.entries["Alcoholism (0/1)"].get())],
                'SMS_received': [int(self.entries["SMS Received (0/1)"].get())],
                'lead_time': [int(self.entries["Lead Time (days)"].get())],
                'Handcap': [int(self.handcap_var.get())],
                'Gender': [self.entries["Gender (M/F)"].get().strip().upper()]
            }

            pred, prob = predict_new_patient(model, input_data, columns)

            show_prob = (1 - prob) * 100
            no_show_prob = prob * 100

            if pred == 1:
                self.result_label.config(text=f"PREDICTION: NO-SHOW\nProbability: {no_show_prob:.2f}%", fg="red")
            else:
                self.result_label.config(text=f"PREDICTION: SHOW UP\nProbability: {show_prob:.2f}%", fg="green")

            self.show_pie_chart([show_prob, no_show_prob], ['Show', 'No-Show'])

        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    # -----------------------------------------------------

    def show_random_data(self):
        sample = display_random_data(self.df)

        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(tk.END, sample.to_string())

        show_percent = (1 - sample['No-show'].mean()) * 100
        no_show_percent = sample['No-show'].mean() * 100

        self.show_pie_chart([show_percent, no_show_percent], ['Show', 'No-Show'])

    # -----------------------------------------------------

    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)
        self.master.geometry('900x700')
        return "break"


# YOUR FILE PATH
file_path = r"C:\Users\Lakshay Bishnoi\OneDrive\Desktop\study\ml project\KaggleV2-May-2016.csv"

df = load_and_prepare_data(file_path)
model, columns = train_model(df)

root = tk.Tk()
app = NoShowPredictorApp(root, df)
root.mainloop()
