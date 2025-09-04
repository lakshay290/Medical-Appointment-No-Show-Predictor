import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import messagebox

def load_and_prepare_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Successfully loaded the dataset.")
        print("Columns in the DataFrame:", df.columns.tolist())
        df.columns = df.columns.str.strip()
        if 'No-show' not in df.columns:
            raise ValueError("The 'No-show' column is missing from the dataset.")
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
        exit()
    except ValueError as ve:
        print(ve)
        exit()
    return df

def train_model(df):
    df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], errors='coerce')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')
    df.dropna(subset=['ScheduledDay', 'AppointmentDay'], inplace=True)
    df['lead_time'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df = df[df['Age'] >= 0]
    df = df[df['lead_time'] >= 0]
    df['has_handicap'] = df['Handcap'].apply(lambda x: 1 if x > 0 else 0)
    features = [
        'Age', 'Scholarship', 'Hipertension', 'Diabetes',
        'Alcoholism', 'SMS_received', 'lead_time', 'has_handicap', 'Gender'
    ]
    X = df[features]
    y = df['No-show']
    X = pd.get_dummies(X, columns=['Gender'], drop_first=True, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_train.columns

def predict_new_patient(model, input_data, columns):
    new_patient_df = pd.DataFrame(input_data)
    new_patient_df = pd.get_dummies(new_patient_df, columns=['Gender'], drop_first=True, dtype=int)
    new_patient_df = new_patient_df.reindex(columns=columns, fill_value=0)
    prediction = model.predict(new_patient_df)
    prediction_proba = model.predict_proba(new_patient_df)
    return prediction[0], prediction_proba[0][1]

def display_random_data(df):
    random_data = df.sample(n=20)
    return random_data

class NoShowPredictorApp:
    def __init__(self, master, df):
        self.master = master
        master.title("No-Show Prediction")
        self.master.attributes('-fullscreen', True)
        self.master.bind("<Escape>", self.exit_fullscreen)
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        title_label = tk.Label(self.main_frame, text="Medical Appointment No-Show Predictor",
                              font=('Helvetica', 18, 'bold'))
        title_label.pack(pady=15)
        self.input_frame = tk.LabelFrame(self.main_frame, text="Patient Information", padx=20, pady=15, font=('Helvetica', 12))
        self.input_frame.pack(fill=tk.X, padx=50, pady=10)
        labels = [
            "Age", "Scholarship (0/1)", "Hypertension (0/1)",
            "Diabetes (0/1)", "Alcoholism (0/1)", "SMS Received (0/1)",
            "Lead Time (days)", "Has Handicap (0/1)", "Gender (M/F)"
        ]
        self.entries = {}
        for i, label_text in enumerate(labels):
            row = i // 3
            col = (i % 3) * 2
            label = tk.Label(self.input_frame, text=label_text, font=('Helvetica', 11))
            label.grid(row=row, column=col, sticky="e", pady=8, padx=5)
            entry = tk.Entry(self.input_frame, width=15, font=('Helvetica', 11))
            entry.grid(row=row, column=col+1, pady=8, padx=5)
            self.entries[label_text] = entry
        button_frame = tk.Frame(self.main_frame)
        button_frame.pack(pady=15)
        self.predict_button = tk.Button(button_frame, text="Predict", command=self.predict,
                                      bg="#4CAF50", fg="white", width=20, font=('Helvetica', 12, 'bold'))
        self.predict_button.grid(row=0, column=0, padx=20)
        self.show_data_button = tk.Button(button_frame, text="Show Random Data", command=self.show_random_data,
                                        bg="#2196F3", fg="white", width=20, font=('Helvetica', 12, 'bold'))
        self.show_data_button.grid(row=0, column=1, padx=20)
        self.exit_button = tk.Button(button_frame, text="Exit Full Screen", command=self.exit_fullscreen,
                                    bg="#f44336", fg="white", width=20, font=('Helvetica', 12, 'bold'))
        self.exit_button.grid(row=0, column=2, padx=20)
        result_frame = tk.LabelFrame(self.main_frame, text="Prediction Result", font=('Helvetica', 12))
        result_frame.pack(fill=tk.X, padx=50, pady=15)
        self.result_label = tk.Label(result_frame, text="No prediction yet", font=('Helvetica', 16, 'bold'), pady=15)
        self.result_label.pack()
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

    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)
        self.master.geometry('800x700')
        return "break"

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
                'has_handicap': [int(self.entries["Has Handicap (0/1)"].get())],
                'Gender': [self.entries["Gender (M/F)"].get().strip().upper()]
            }
            for key, value in input_data.items():
                if key != "Gender":
                    if value[0] < 0:
                        raise ValueError(f"{key} cannot be negative")
                else:
                    if value[0] not in ['M', 'F']:
                        raise ValueError("Gender must be 'M' or 'F'")
            prediction, probability = predict_new_patient(model, input_data, columns)
            if prediction == 1:
                result_text = "PREDICTION: NO-SHOW\n"
                result_text += f"Probability of missing appointment: {probability:.2%}"
                color = "red"
            else:
                result_text = "PREDICTION: SHOW UP\n"
                result_text += f"Probability of attending appointment: {(1-probability):.2%}"
                color = "green"
            self.result_label.config(text=result_text, fg=color)
        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    def show_random_data(self):
        random_data = display_random_data(self.df)
        data_text = random_data.to_string()
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(tk.END, data_text)

file_path = r"C:\Users\laksh\Desktop\KaggleV2-May-2016.csv"
df = load_and_prepare_data(file_path)
model, columns = train_model(df)
root = tk.Tk()
app = NoShowPredictorApp(root, df)
root.mainloop()