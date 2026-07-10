import os

print("\n--- 🔍 FOLDER DIAGNOSTIC ---")
print(f"Current Directory: {os.getcwd()}")

if os.path.exists("databases"):
    folders = os.listdir("databases")
    print(f"✅ 'databases' folder FOUND!")
    print(f"📂 Contents inside: {folders}")
    if not folders:
        print("❌ WARNING: The databases folder is completely EMPTY.")
else:
    print("❌ CRITICAL ERROR: 'databases' folder DOES NOT EXIST in this directory.")
    print("💡 Fix: Make sure your databases folder is here, and you are not running the terminal from inside the 'src' folder.")
print("----------------------------\n")