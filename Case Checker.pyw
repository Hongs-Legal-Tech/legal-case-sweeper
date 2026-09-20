import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser
import re
from pathlib import Path
from typing import Dict, Callable, List, Optional, Any

import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError
import openpyxl
from openpyxl.utils.exceptions import InvalidFileException

# LLM SDKs
from google import genai
from google.genai import errors as genai_errors
import openai
import anthropic

MAX_TOKEN_CHAR_LIMIT = 200000
EXCEL_SAVE_INTERVAL = 10


class LLMClientAdapter:
    """
    Unified interface for interacting with multiple LLM providers.
    Implements the Adapter pattern to standardize API calls and error handling.
    """
    def __init__(self, provider: str, api_key: str, model_id: str, base_url: str = "") -> None:
        self.provider = provider
        self.model_id = model_id
        self.client: Any = None

        if self.provider == "Google":
            self.client = genai.Client(api_key=api_key)
        elif self.provider == "OpenAI":
            self.client = openai.OpenAI(api_key=api_key)
        elif self.provider == "Anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "Custom/Local (OpenAI-Compatible)":
            self.client = openai.OpenAI(
                api_key=api_key if api_key else "not-needed", 
                base_url=base_url if base_url else "http://localhost:11434/v1"
            )

    def generate(self, prompt: str) -> str:
        """Routes the prompt to the correct SDK and standardizes the response."""
        if self.provider == "Google":
            response = self.client.models.generate_content(
                model=self.model_id, 
                contents=prompt
            )
            return response.text.strip() if response.text else "API_ERROR: Empty Response"
            
        elif self.provider in ["OpenAI", "Custom/Local (OpenAI-Compatible)"]:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = response.choices[0].message.content
            return content.strip() if content else "API_ERROR: Empty Response"
            
        elif self.provider == "Anthropic":
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            if response.content and len(response.content) > 0:
                return response.content[0].text.strip()
            return "API_ERROR: Empty Response"
            
        raise ValueError(f"Unsupported provider: {self.provider}")


class CaseAnalyzer:
    """
    Handles PDF extraction, LLM API communication, and Excel exporting.
    """
    def __init__(
        self, 
        provider: str,
        api_key: str, 
        model_id: str,
        base_url: str,
        target_dir: Path, 
        out_excel: Path, 
        user_query: str, 
        update_callbacks: Dict[str, Callable]
    ) -> None:
        self.target_dir = target_dir
        self.out_excel = out_excel
        self.user_query = user_query
        self.callbacks = update_callbacks
        self.cancel_event = threading.Event()
        
        self.llm = LLMClientAdapter(provider, api_key, model_id, base_url)

    def cancel(self) -> None:
        self.cancel_event.set()

    def extract_pdf_text(self, pdf_path: Path) -> str:
        text_blocks: List[str] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    if self.cancel_event.is_set():
                        return ""
                    page_text = page.extract_text()
                    if page_text:  
                        text_blocks.append(page_text)
            return "\n".join(text_blocks)
        except (PDFSyntaxError, ValueError, TypeError) as e:
            return f"EXTRACTION_ERROR: Corrupt or unreadable PDF - {str(e)}"
        except OSError as e:
            return f"EXTRACTION_ERROR: File system error - {str(e)}"

    def _call_api_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        backoff_factor = 2
        for attempt in range(max_retries):
            if self.cancel_event.is_set():
                return "Cancelled"
            try:
                return self.llm.generate(prompt)
            except (ConnectionError, TimeoutError) as e:
                if attempt == max_retries - 1:
                    return f"API_ERROR: Network failure: {str(e)}"
                time.sleep(backoff_factor ** attempt)
            except (genai_errors.APIError, openai.APIError, anthropic.APIError) as e:
                if attempt == max_retries - 1:
                    return f"API_ERROR: Provider Error: {str(e)}"
                time.sleep(backoff_factor ** attempt)
            except Exception as e:
                return f"API_ERROR: Unexpected failure: {str(e)}"
        return "API_ERROR: Unknown failure"

    def _safe_save_excel(self, wb: openpyxl.Workbook) -> None:
        try:
            wb.save(self.out_excel)
        except PermissionError:
            self.callbacks['status']("WARNING: Excel file is open. Close it to save progress!")
        except (OSError, InvalidFileException) as e:
            self.callbacks['status'](f"Excel Save Error: {str(e)}")

    def run(self) -> None:
        try:
            self.callbacks['status']("Indexing PDFs...")
            pdf_files = list(self.target_dir.rglob('*.pdf'))
            total_files = len(pdf_files)
            
            if total_files == 0:
                self.callbacks['status']("No PDFs found in the target directory.")
                return

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Case Law Analysis"
            ws.append(["File Name", "File Path", "Is Opinion (Yes/No)", "Topically Relevant (Yes/No)", "Case Summary"])

            for idx, pdf_path in enumerate(pdf_files):
                if self.cancel_event.is_set():
                    self.callbacks['status']("Process Killed by User.")
                    break
                
                filename = pdf_path.name
                self.callbacks['status'](f"Processing {idx+1}/{total_files}: {filename}")
                self.callbacks['progress'](idx, total_files)
                
                text = self.extract_pdf_text(pdf_path)
                
                if text.startswith("EXTRACTION_ERROR") or not text.strip():
                    error_msg = text if text.startswith("EXTRACTION_ERROR") else "No Text Found in PDF"
                    ws.append([filename, str(pdf_path), "Error", "Error", error_msg])
                    continue
                
                text_to_analyze = text[:MAX_TOKEN_CHAR_LIMIT] 
                
                # STAGE 1: OPINION FILTER
                prompt_1 = (
                    "Review the following legal document. Determine if it is an actual, substantive court opinion or decision on the merits. "
                    "You must classify it as NOT an opinion if it is merely a denial of a writ, an administrative order, a public statement, a routine procedural filing, or a non-precedential scheduling order.\n\n"
                    "Format your response EXACTLY as follows:\nIs Opinion: [Yes or No]\n\n"
                    f"Document Text:\n{text_to_analyze}"
                )
                
                stage_1_result = self._call_api_with_retry(prompt_1)
                if self.cancel_event.is_set(): break

                is_opinion_val = "Unknown"
                op_match = re.search(r'Is Opinion:\s*(Yes|No)', stage_1_result, re.IGNORECASE)
                if op_match:
                    is_opinion_val = op_match.group(1).capitalize()
                elif "yes" in stage_1_result.lower():
                    is_opinion_val = "Yes"
                elif "no" in stage_1_result.lower():
                    is_opinion_val = "No"

                if is_opinion_val == "No":
                    ws.append([filename, str(pdf_path), "No", "Skipped", "Skipped (Not a substantive opinion)"])
                    if idx % EXCEL_SAVE_INTERVAL == 0:
                        self._safe_save_excel(wb)
                    continue

                # STAGE 2: TOPICAL FILTER
                topical_val = "N/A (No Query)"
                if self.user_query:
                    prompt_2 = (
                        f"Review the following legal document against this specific query/topic: \"{self.user_query}\"\n\n"
                        "Determine if the substantive facts or legal analysis in the document are relevant to this query.\n\n"
                        "Format your response EXACTLY as follows:\nTopically Relevant: [Yes or No]\n\n"
                        f"Document Text:\n{text_to_analyze}"
                    )
                    
                    stage_2_result = self._call_api_with_retry(prompt_2)
                    if self.cancel_event.is_set(): break
                    
                    top_match = re.search(r'Topically Relevant:\s*(Yes|No)', stage_2_result, re.IGNORECASE)
                    if top_match:
                        topical_val = top_match.group(1).capitalize()
                    elif "yes" in stage_2_result.lower():
                        topical_val = "Yes"
                    elif "no" in stage_2_result.lower():
                        topical_val = "No"
                    else:
                        topical_val = "Unknown"

                    if topical_val == "No":
                        ws.append([filename, str(pdf_path), "Yes", "No", "Skipped (Not relevant to query)"])
                        if idx % EXCEL_SAVE_INTERVAL == 0:
                            self._safe_save_excel(wb)
                        continue

                # STAGE 3: STRICT SUMMARIZER
                prompt_3 = (
                    "Summarize the following legal court opinion.\n\n"
                    "STRICT OUTPUT CONSTRAINTS:\n"
                    "1. NO markdown, NO headings, NO bullet points, NO conversational filler.\n"
                    "2. Write EXACTLY 3 sentences in a single paragraph:\n"
                    "   - Sentence 1: The core facts of the case.\n"
                    "   - Sentence 2: The court's primary legal analysis.\n"
                    "   - Sentence 3: The final decision or holding.\n"
                    "3. Absolute maximum length: 100 words.\n\n"
                    f"Document Text:\n{text_to_analyze}"
                )
                stage_3_result = self._call_api_with_retry(prompt_3)
                if self.cancel_event.is_set(): break
                
                stage_3_result = stage_3_result.replace('\n', ' ').strip()
                ws.append([filename, str(pdf_path), is_opinion_val, topical_val, stage_3_result])
                
                if idx % EXCEL_SAVE_INTERVAL == 0:
                    self._safe_save_excel(wb)

            if not self.cancel_event.is_set():
                self._safe_save_excel(wb)
                self.callbacks['progress'](total_files, total_files)
                self.callbacks['status'](f"Complete. Saved to {self.out_excel.name}")
            
        except Exception as e:
            self.callbacks['status'](f"Fatal Pipeline Error: {str(e)}")
            
        finally:
            self.callbacks['done']()


class AICaseAnalyzerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Legal Case Sweeper")
        self.root.geometry("680x780") 
        self.root.resizable(False, False)
        
        self.bg_color = "#1e1e1e"
        self.fg_color = "#d4d4d4"
        self.entry_bg = "#2d2d2d"
        self.btn_bg = "#3d3d3d"
        self.btn_active = "#5d5d5d"
        
        self.root.configure(bg=self.bg_color)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Horizontal.TProgressbar", background="#2f855a", troughcolor=self.entry_bg, bordercolor=self.bg_color)
        style.configure("TCombobox", fieldbackground=self.entry_bg, background=self.btn_bg, foreground=self.fg_color)

        lbl_args = {"bg": self.bg_color, "fg": self.fg_color, "font": ("Arial", 10)}
        ent_args = {"bg": self.entry_bg, "fg": self.fg_color, "insertbackground": self.fg_color, "highlightthickness": 1, "highlightbackground": "#444", "relief": "flat"}
        btn_args = {"bg": self.btn_bg, "fg": self.fg_color, "activebackground": self.btn_active, "activeforeground": "white", "relief": "flat", "font": ("Arial", 9)}

        self.analyzer: Optional[CaseAnalyzer] = None
        
        # Row 0: Provider Selection
        tk.Label(root, text="AI Provider:", **lbl_args).grid(row=0, column=0, padx=10, pady=(20, 10), sticky="e")
        self.provider_var = tk.StringVar(value="Google")
        self.provider_combo = ttk.Combobox(root, textvariable=self.provider_var, state="readonly", width=48, values=[
            "Google", "OpenAI", "Anthropic", "Custom/Local (OpenAI-Compatible)"
        ])
        self.provider_combo.grid(row=0, column=1, padx=5, pady=(20, 10), sticky="w")
        self.provider_combo.bind("<<ComboboxSelected>>", self.on_provider_change)

        # Row 1: API Key
        tk.Label(root, text="API Key:", **lbl_args).grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.api_var = tk.StringVar()
        self.api_entry = tk.Entry(root, textvariable=self.api_var, width=50, show="*", **ent_args)
        self.api_entry.grid(row=1, column=1, padx=5, pady=10, ipady=3)
        
        # Row 2: Base URL (For Custom/Local)
        tk.Label(root, text="Base URL (Local/Custom):", **lbl_args).grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.url_var = tk.StringVar()
        self.url_entry = tk.Entry(root, textvariable=self.url_var, width=50, state="disabled", **ent_args)
        self.url_entry.grid(row=2, column=1, padx=5, pady=10, ipady=3)

        # Row 3: Model Selection
        tk.Label(root, text="AI Model:", **lbl_args).grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(root, textvariable=self.model_var, width=48)
        self.on_provider_change() 
        self.model_combo.grid(row=3, column=1, padx=5, pady=10, sticky="w")
        
        # Row 4: Target Folder
        tk.Label(root, text="Target Folder:", **lbl_args).grid(row=4, column=0, padx=10, pady=10, sticky="e")
        self.dir_var = tk.StringVar()
        tk.Entry(root, textvariable=self.dir_var, width=50, **ent_args).grid(row=4, column=1, padx=5, pady=10, ipady=3)
        tk.Button(root, text="Browse", command=self.browse_dir, **btn_args).grid(row=4, column=2, padx=5, pady=10, ipadx=5)
        
        # Row 5: Output Excel
        tk.Label(root, text="Output Excel:", **lbl_args).grid(row=5, column=0, padx=10, pady=10, sticky="e")
        self.out_var = tk.StringVar()
        tk.Entry(root, textvariable=self.out_var, width=50, **ent_args).grid(row=5, column=1, padx=5, pady=10, ipady=3)
        tk.Button(root, text="Browse", command=self.browse_out, **btn_args).grid(row=5, column=2, padx=5, pady=10, ipadx=5)
        
        # Row 6: Topical Query
        tk.Label(root, text="Topical Query\n(Optional):", **lbl_args).grid(row=6, column=0, padx=10, pady=10, sticky="ne")
        query_frame = tk.Frame(root, bg=self.bg_color)
        query_frame.grid(row=6, column=1, columnspan=2, padx=5, pady=10, sticky="w")
        
        self.query_text = tk.Text(query_frame, height=6, width=48, bg=self.entry_bg, fg=self.fg_color, 
                                  insertbackground=self.fg_color, highlightthickness=1, 
                                  highlightbackground="#444", relief="flat", font=("Arial", 10), wrap="word")
        self.query_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(query_frame, command=self.query_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.query_text.config(yscrollcommand=scrollbar.set)
        
        # Row 7: Progress Bar
        self.progress = ttk.Progressbar(root, orient="horizontal", length=630, mode="determinate", style="Horizontal.TProgressbar")
        self.progress.grid(row=7, column=0, columnspan=3, padx=10, pady=20)
        
        # Row 8: Status
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(root, textvariable=self.status_var, bg=self.bg_color, fg="#63b3ed", font=("Arial", 10, "italic")).grid(row=8, column=0, columnspan=3)
        
        # Row 9: Buttons
        btn_frame = tk.Frame(root, bg=self.bg_color)
        btn_frame.grid(row=9, column=0, columnspan=3, pady=15)
        
        self.start_btn = tk.Button(btn_frame, text="START SWEEP", command=self.start_processing, 
                                   bg="#2f855a", fg="white", font=("Arial", 10, "bold"), 
                                   activebackground="#276749", activeforeground="white", relief="flat", width=20)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.kill_btn = tk.Button(btn_frame, text="KILL PROCESS", command=self.kill_process, 
                                  bg="#c53030", fg="white", font=("Arial", 10, "bold"), state="disabled",
                                  activebackground="#9b2c2c", activeforeground="white", relief="flat", width=20)
        self.kill_btn.grid(row=0, column=1, padx=10)

        # Row 10: Footer
        self.footer_label = tk.Label(
            root, 
            text="Copyright Al H. Health Lawyer, New-Tech Analyst | Hongs-Legal-Tec | https://github.com/Hongs-Legal-Tech", 
            bg=self.bg_color, fg="#888888", font=("Arial", 9), cursor="hand2"
        )
        self.footer_label.grid(row=10, column=0, columnspan=3, pady=(15, 10))
        self.footer_label.bind("<Button-1>", lambda e: webbrowser.open_new("https://github.com/Hongs-Legal-Tech"))

    def on_provider_change(self, event=None) -> None:
        """Updates the model combobox defaults and toggles the Base URL field based on provider."""
        provider = self.provider_var.get()
        self.url_entry.config(state="disabled")
        
        if provider == "Google":
            models = [
                "gemini-3.8-flash", 
                "gemini-3.1-pro", 
                "gemini-3-pro-deep-think", 
                "gemma-4-26b-a4b"
            ]
            self.model_var.set(models[0])
            self.model_combo['values'] = models
            
        elif provider == "OpenAI":
            models = [
                "gpt-6-astra", 
                "gpt-5.6-sol", 
                "gpt-5.4", 
                "gpt-oss-120b"
            ]
            self.model_var.set(models[0])
            self.model_combo['values'] = models
            
        elif provider == "Anthropic":
            models = [
                "claude-5.1-fable-latest", 
                "claude-5-opus-latest", 
                "claude-5-sonnet-latest",
                "claude-4.6-sonnet-latest"
            ]
            self.model_var.set(models[0])
            self.model_combo['values'] = models
            
        elif provider == "Custom/Local (OpenAI-Compatible)":
            self.url_entry.config(state="normal")
            if not self.url_var.get():
                self.url_var.set("http://localhost:11434/v1") 
            models = [
                "gpt-oss-120b",
                "gemma-4-26b-a4b",
                "llama-3.3-70b", 
                "deepseek-r1"
            ]
            self.model_var.set(models[0])
            self.model_combo['values'] = models

    def browse_dir(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.dir_var.set(folder)
            
    def browse_out(self) -> None:
        file = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if file:
            self.out_var.set(file)

    def update_status(self, msg: str) -> None:
        self.root.after(0, lambda: self.status_var.set(msg))

    def update_progress(self, current: int, total: int) -> None:
        def set_prog():
            self.progress["maximum"] = total
            self.progress["value"] = current
        self.root.after(0, set_prog)

    def reset_buttons(self) -> None:
        self.root.after(0, lambda: self.start_btn.config(state="normal"))
        self.root.after(0, lambda: self.kill_btn.config(state="disabled"))

    def kill_process(self) -> None:
        if messagebox.askyesno("Confirm Kill", "Terminate processing?"):
            if self.analyzer:
                self.analyzer.cancel()
            self.update_status("Cancelling... Please wait for the current operation to finish.")
            self.kill_btn.config(state="disabled")

    def start_processing(self) -> None:
        provider = self.provider_var.get()
        api_key = self.api_var.get().strip()
        model_id = self.model_var.get().strip()
        base_url = self.url_var.get().strip()
        target_dir_str = self.dir_var.get().strip()
        out_excel_str = self.out_var.get().strip()
        user_query = self.query_text.get("1.0", "end-1c").strip()
        
        if provider != "Custom/Local (OpenAI-Compatible)" and not api_key:
            messagebox.showerror("API Key Error", f"Please provide a valid API Key for {provider}.")
            return

        if not model_id:
            messagebox.showerror("Model Error", "Please specify an AI Model.")
            return

        if not target_dir_str or not out_excel_str:
            messagebox.showerror("Error", "Target Directory and Output Excel fields are required.")
            return

        target_dir = Path(target_dir_str)
        out_excel = Path(out_excel_str)

        if not target_dir.is_dir():
            messagebox.showerror("Path Error", "The specified Target Directory does not exist or is invalid.")
            return
            
        if not out_excel.parent.exists():
            messagebox.showerror("Path Error", "The output directory for the Excel file does not exist.")
            return

        self.start_btn.config(state="disabled")
        self.kill_btn.config(state="normal")
        self.progress["value"] = 0
        
        callbacks = {
            'status': self.update_status,
            'progress': self.update_progress,
            'done': self.reset_buttons
        }
        
        self.analyzer = CaseAnalyzer(provider, api_key, model_id, base_url, target_dir, out_excel, user_query, callbacks)
        threading.Thread(target=self.analyzer.run, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = AICaseAnalyzerGUI(root)
    root.mainloop()