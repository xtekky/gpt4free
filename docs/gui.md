# G4F - GUI Documentation

## Overview
The G4F GUI is a self-contained, user-friendly interface designed for interacting with multiple AI models from various providers. It allows users to generate text, code, and images effortlessly. Advanced features such as speech recognition, file uploads, conversation backup/restore, and more are included. Both the backend and frontend are fully integrated into the GUI, making setup simple and seamless.

## Features

### 1. **Multiple Providers and Models**
   - **Provider/Model Selection via Dropdown:** Use the select box to choose a specific **provider/model combination**. 
   - **Pinning Provider/Model Combinations:** After selecting a provider and model from the dropdown, click the **pin button** to add the combination to the pinned list.
   - **Remove Pinned Combinations:** Each pinned provider/model combination is displayed as a button. Clicking on the button removes it from the pinned list.
   - **Send Requests to Multiple Providers:** You can pin multiple provider/model combinations and send requests to all of them simultaneously, enabling fast and comprehensive content generation.

### 2. **Text, Code, and Image Generation**
   - **Text and Code Generation:** Enter prompts to generate text or code outputs.
   - **Image Generation:** Provide text prompts to generate images, which are shown as thumbnails. Clicking on a thumbnail opens the image in a lightbox view.

### 3. **Gallery Functionality**
   - **Image Thumbnails:** Generated images appear as small thumbnails within the conversation.
   - **Lightbox View:** Clicking a thumbnail opens the image in full size, along with the prompt used to generate it.
   - **Automatic Image Download:** Enable automatic downloading of generated images through the settings.

### 4. **Conversation Management**
   - **Message Reuse:** While messages can't be edited, you can copy and reuse them.
   - **Message Deletion:** Conversations can be deleted for a cleaner workspace.
   - **Conversation List:** The left sidebar displays a list of active and past conversations for easy navigation.
   - **Change Conversation Title:** By clicking the three dots next to a conversation title, you can either delete or change its title.
   - **Backup and Restore Conversations:** Backup and restore all conversations and messages as a JSON file (accessible via the settings).

### 5. **Speech Recognition and Synthesis**
   - **Speech Input:** Use speech recognition to input prompts by speaking instead of typing.
   - **Speech Output (Text-to-Speech):** The generated text can be read aloud using speech synthesis.
   - **Custom Language Settings:** Configure the language used for speech recognition to match your preference.

### 6. **File Uploads**
   - **Image Uploads:** Upload images that will be appended to your message and sent to the AI provider.
   - **Text File Uploads:** Upload text files, and their contents will be added to the message to provide more detailed input to the AI.

### 7. **Web Access and Settings**
   - **DuckDuckGo Web Access:** Enable web access through DuckDuckGo for privacy-focused browsing.
   - **Theme Toggle:** Switch between **dark mode** and **light mode** in the settings.
   - **Provider Visibility:** Hide unused providers in the settings using toggle buttons.
   - **Log Access:** View application logs, including error messages and debug logs, through the settings.

### 8. **Authentication**
   - **Basic Authentication:** Set a password for Basic Authentication using the `--g4f-api-key` argument when starting the web server.

## Installation

You can install the G4F GUI either as a full stack or in a lightweight version:

1. **Full Stack Installation** (includes all packages, including browser support and drivers):
   ```bash
   pip install -U g4f[all]
   ```

2. **Slim Installation** (does not include browser drivers, suitable for headless environments):
   ```bash
   pip install -U g4f[slim]
   ```

   - **Full Stack Installation:** This installs all necessary dependencies, including browser support for web-based interactions.
   - **Slim Installation:** This version is lighter, with no browser support, ideal for environments where browser interactions are not required.

## Setup

### Setting the Environment Variable

It is **recommended** to set a `G4F_API_KEY` environment variable for authentication. You can do this as follows:

On **Linux/macOS**:
```bash
export G4F_API_KEY="your-api-key-here"
```

On **Windows**:
```bash
set G4F_API_KEY="your-api-key-here"
```

### Start the GUI and Backend

Run the following command to start both the GUI and backend services based on the G4F client:

```bash
python -m g4f --debug --port 8080 --g4f-api-key $G4F_API_KEY
```

This starts the GUI at `http://localhost:8080` with all necessary backend components running seamlessly.

### Access the GUI

Once the server is running, open your browser and navigate to:

```
http://localhost:8080/chat/
```

## Using the Interface

1. **Select and Manage Providers/Models:**
   - Use the **select box** to view the list of available providers and models.
   - Select a **provider/model combination** from the dropdown.
   - Click the **pin button** to add the combination to your pinned list.
   - To **unpin** a combination, click the corresponding button in the pinned list.

2. **Input a Prompt:**
   - Enter your prompt manually or use **speech recognition** to dictate it.
   - You can also upload **images** or **text files** to be included in the prompt.

3. **Generate Content:**
   - Click the "Generate" button to produce the content.
   - The AI will generate text, code, or images depending on the prompt.

4. **View and Interact with Results:**
   - **For Text/Code:** The generated content will appear in the conversation window.
   - **For Images:** Generated images will be shown as thumbnails. Click on them to view in full size.

5. **Backup and Restore Conversations:**
   - Backup all your conversations as a **JSON file** and restore them at any time via the settings.

6. **Manage Conversations:**
   - Delete or rename any conversation by clicking the three dots next to the conversation title.

### Gallery Functionality

- **Image Thumbnails:** All generated images are shown as thumbnails within the conversation window.
- **Lightbox View:** Clicking a thumbnail opens the image in a larger view along with the associated prompt.
- **Automatic Image Download:** Enable automatic downloading of generated images in the settings.

## Settings Configuration

1. **API Key:** Set your API key when starting the server by defining the `G4F_API_KEY` environment variable.
2. **Provider Visibility:** Hide unused providers through the settings.
3. **Theme:** Toggle between **dark mode** and **light mode**. Disabling dark mode switches to a white theme.
4. **DuckDuckGo Access:** Enable DuckDuckGo for privacy-focused web browsing.
5. **Speech Recognition Language:** Set your preferred language for speech recognition.
6. **Log Access:** View logs, including error and debug messages, from the settings menu.
7. **Automatic Image Download:** Enable this feature to automatically download generated images.

## Known Issues

- **Gallery Loading:** Large images may take time to load depending on system performance.
- **Speech Recognition Accuracy:** Accuracy may vary depending on microphone quality or speech clarity.
- **Provider Downtime:** Some AI providers may experience downtime or disruptions.

[Return to Home](/)