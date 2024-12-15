import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

class AutoMLGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AutoML GUI")

        # Initialize data variables
        self.data = None
        self.target_column = None

        # GUI Components
        self.create_widgets()

    def create_widgets(self):
        """Create the main interface elements."""
        tk.Button(self.root, text="Load Dataset", command=self.load_dataset).pack(pady=10)
        
        self.dataset_label = tk.Label(self.root, text="No dataset loaded", fg="red")
        self.dataset_label.pack()

        tk.Label(self.root, text="Target Column:").pack(pady=5)
        self.target_entry = tk.Entry(self.root)
        self.target_entry.pack()

        tk.Label(self.root, text="Task Type (Classification/Regression):").pack(pady=5)
        self.task_type_entry = tk.Entry(self.root)
        self.task_type_entry.pack()

        tk.Button(self.root, text="Run AutoML", command=self.run_automl).pack(pady=10)
        tk.Button(self.root, text="Visualize Data", command=self.visualize_data).pack(pady=10)

        self.result_text = tk.Text(self.root, height=15, width=70)
        self.result_text.pack(pady=10)

    def load_dataset(self):
        """Load a dataset file."""
        file_path = filedialog.askopenfilename(filetypes=[["CSV Files", "*.csv"]])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.dataset_label.config(text=f"Loaded: {os.path.basename(file_path)}", fg="green")
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def visualize_data(self):
        """Visualize the dataset."""
        if self.data is None:
            messagebox.showwarning("No Dataset", "Please load a dataset first.")
            return

        try:
            plt.figure(figsize=(10, 6))
            self.data.hist(bins=30, figsize=(12, 8))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {e}")

    def run_automl(self):
        """Perform basic AutoML operations."""
        if self.data is None:
            messagebox.showwarning("No Dataset", "Please load a dataset first.")
            return

        self.target_column = self.target_entry.get()
        if self.target_column not in self.data.columns:
            messagebox.showerror("Error", "Target column not found in dataset.")
            return

        task_type = self.task_type_entry.get().strip().lower()
        if task_type not in ["classification", "regression"]:
            messagebox.showerror("Error", "Invalid task type. Enter 'Classification' or 'Regression'.")
            return

        try:
            # Preprocessing
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Running AutoML...\n")

            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]

            # Handle categorical data
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if task_type == "classification":
                # Model selection
                model = GradientBoostingClassifier(random_state=42)
            else:
                # Regression (placeholder model)
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=42)

            # GridSearch for hyperparameter tuning (example for classification)
            if task_type == "classification":
                param_grid = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 5, 7]
                }
                grid_search = GridSearchCV(model, param_grid, cv=3)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            if task_type == "classification":
                acc = accuracy_score(y_test, y_pred)
                self.result_text.insert(tk.END, f"Accuracy: {acc:.2f}\n")
                self.result_text.insert(tk.END, "Classification Report:\n")
                self.result_text.insert(tk.END, classification_report(y_test, y_pred))
            else:
                from sklearn.metrics import mean_squared_error
                mse = mean_squared_error(y_test, y_pred)
                self.result_text.insert(tk.END, f"Mean Squared Error: {mse:.2f}\n")

            messagebox.showinfo("Success", "AutoML completed!")
        except Exception as e:
            messagebox.showerror("Error", f"AutoML failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoMLGUI(root)
    root.mainloop()