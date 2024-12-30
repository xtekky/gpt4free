# G4F - GUI Documentation

## Overview
The G4F GUI is a self-contained, user-friendly interface designed for interacting with multiple AI models from various providers. It allows users to generate text, code, and images effortlessly. Advanced features such as speech recognition, file uploads, conversation backup/restore, and more are included. Both the backend and frontend are fully integrated into the GUI, making setup simple and seamless.

## Features

### 1. **Multiple Providers and Models**
   - **Provider/Model Selection via Dropdown**  
     Use the select box to choose a specific **provider/model combination**.  
   - **Pinning Provider/Model Combinations**  
     After selecting a provider and model from the dropdown, click the **pin button** to add the combination to the pinned list.  
   - **Remove Pinned Combinations**  
     Each pinned provider/model combination is displayed as a button. Clicking on the button removes it from the pinned list.  
   - **Send Requests to Multiple Providers**  
     You can pin multiple provider/model combinations and send requests to all of them simultaneously, enabling fast and comprehensive content generation.

### 2. **Text, Code, and Image Generation**
   - **Text and Code Generation**  
     Enter prompts to generate text or code outputs.  
   - **Image Generation**  
     Provide text prompts to generate images, which are shown as thumbnails. Clicking on a thumbnail opens the image in a lightbox view.

### 3. **Gallery Functionality**
   - **Image Thumbnails**  
     Generated images appear as small thumbnails within the conversation.  
   - **Lightbox View**  
     Clicking a thumbnail opens the image in full size, along with the prompt used to generate it.  
   - **Automatic Image Download**  
     You can enable automatic downloading of generated images through the settings.

### 4. **Conversation Management**
   - **Message Reuse**  
     While messages cannot be edited after sending, you can copy and reuse them.  
   - **Message Deletion**  
     Individual messages or entire conversations can be deleted for a cleaner workspace.  
   - **Conversation List**  
     The left sidebar displays a list of active and past conversations for easy navigation.  
   - **Change Conversation Title**  
     By clicking the three dots next to a conversation title, you can either delete the conversation or change its title.  
   - **Backup and Restore Conversations**  
     Backup and restore all conversations and messages as a JSON file (accessible via the settings).

### 5. **Speech Recognition and Synthesis**
   - **Speech Input**  
     Use speech recognition to input prompts by speaking instead of typing.  
   - **Speech Output (Text-to-Speech)**  
     The generated text can be read aloud using speech synthesis.  
   - **Custom Language Settings**  
     Configure the language used for speech recognition to match your preference.

### 6. **File Uploads**
   - **Image Uploads**  
     Upload images that will be appended to your message and sent to the AI provider.  
   - **Text File Uploads**  
     Upload text files; their contents will be added to the message to provide more detailed input to the AI.

### 7. **Web Access and Settings**
   - **DuckDuckGo Web Access**  
     Enable web access through DuckDuckGo for privacy-focused browsing.  
   - **Theme Toggle**  
     Switch between **dark mode** and **light mode** in the settings.  
   - **Provider Visibility**  
     Hide unused providers in the settings using toggle buttons.  
   - **Log Access**  
     View application logs, including error messages and debug logs, through the settings.

### 8. **Authentication**
   - **Basic Authentication**  
     You can set a password for Basic Authentication using the `--g4f-api-key` argument when starting the web server.

### 9. **Continue Button (ChatGPT & HuggingChat)**
   - **Automatic Detection of Truncated Responses**  
     When using **ChatGPT** or **HuggingChat** providers, responses may occasionally be cut off or truncated.  
   - **Continue Button**  
     If the GUI detects that the response ended abruptly, a **Continue** button appears directly below the truncated message. Clicking this button sends a follow-up request to the same provider and model, retrieving the rest of the message.  
   - **Seamless Conversation Flow**  
     This feature ensures that you can read complete messages without manually re-prompting.  

---

## Installation

You can install the G4F GUI either as a full stack or in a lightweight version:

1. **Full Stack Installation** (includes all packages, including browser support and drivers):
   ```bash
   pip install -U g4f[all]
   ```
   
   - Installs all necessary dependencies, including browser support for web-based interactions.

2. **Slim Installation** (does not include browser drivers, suitable for headless environments):
   ```bash
   pip install -U g4f[slim]
   ```
   
   - This version is lighter, with no browser support, ideal for environments where browser interactions are not required.

---

## Setup

### 1. Setting the Environment Variable

It is **recommended** to set a `G4F_API_KEY` environment variable for authentication. You can do this as follows:

- **Linux/macOS**:
  ```bash
  export G4F_API_KEY="your-api-key-here"
  ```

- **Windows**:
  ```bash
  set G4F_API_KEY="your-api-key-here"
  ```

### 2. Start the GUI and Backend

Run the following command to start both the GUI and backend services based on the G4F client:

```bash
python -m g4f --debug --port 8080 --g4f-api-key $G4F_API_KEY
```

This starts the GUI at `http://localhost:8080` with all necessary backend components running seamlessly.

### 3. Access the GUI

Once the server is running, open your browser and navigate to:

```
http://localhost:8080/chat/
```

---

## Using the Interface

1. **Select and Manage Providers/Models**  
   - Use the **select box** to view the list of available providers and models.  
   - Select a **provider/model combination** from the dropdown.  
   - Click the **pin button** to add the combination to your pinned list.  
   - To **unpin** a combination, click the corresponding pinned button.

2. **Input a Prompt**  
   - Enter your prompt manually or use **speech recognition** to dictate it.  
   - You can also upload **images** or **text files** to include them in your prompt.

3. **Generate Content**  
   - Click the **Generate** button to produce the text, code, or images requested.

4. **View and Interact with Results**  
   - **Text/Code:** The generated response appears in the conversation window.  
   - **Images:** Generated images are displayed as thumbnails. Click on any thumbnail to view it in full size within the lightbox.

5. **Continue Button (ChatGPT & HuggingChat)**  
   - If a response is truncated, a **Continue** button will appear under the last message. Clicking it asks the same provider to continue the response from where it ended.

6. **Manage Conversations**  
   - **Delete** or **rename** any conversation by clicking the three dots next to its title.  
   - **Backup/Restore** all your conversations as a JSON file in the settings.

---

## Gallery Functionality

- **Image Thumbnails:** All generated images are shown as thumbnails within the conversation window.  
- **Lightbox View:** Clicking any thumbnail opens the image in a larger view along with the associated prompt.  
- **Automatic Image Download:** Enable this feature in the settings if you want images to be saved automatically.

---

## Settings Configuration

1. **API Key**  
   Set your API key when starting the server by defining the `G4F_API_KEY` environment variable.

2. **Provider Visibility**  
   Hide any providers you don’t plan to use through the settings.

3. **Theme**  
   Toggle between **dark mode** and **light mode**. Disabling dark mode switches to a white theme.

4. **DuckDuckGo Access**  
   Optionally enable DuckDuckGo for privacy-focused web searching.

5. **Speech Recognition Language**  
   Configure your preferred speech recognition language.

6. **Log Access**  
   View logs (including error and debug messages) from the settings menu.

7. **Automatic Image Download**  
   Enable this to have all generated images downloaded immediately upon creation.

---

## Known Issues

1. **Gallery Loading**  
   Large images may take additional time to load depending on your hardware and network.

2. **Speech Recognition Accuracy**  
   Voice recognition may vary with microphone quality, background noise, or speech clarity.

3. **Provider Downtime**  
   Some AI providers may experience temporary downtime or disruptions.

---

[Return to Home](/)