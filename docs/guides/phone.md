### Guide: Running the G4F GUI on Your Smartphone

Running Python applications on your smartphone is possible with specialized apps like Pydroid. This tutorial will walk you through the process using an Android smartphone with Pydroid. Note that the steps may vary slightly for iPhone users due to differences in app names and ownership.

<p align="center">
    On the first screenshot is <strong>Pydroid</strong> and on the second is the <strong>Web UI</strong> in a browser
</p>

<p align="center">
    <img src="/docs/guides/phone.png" />
    <img src="/docs/guides/phone2.jpeg" />
</p>

1. **Install Pydroid from the Google Play Store:**
   - Navigate to the Google Play Store and search for "Pydroid 3 - IDE for Python 3" or use the following link: [Pydroid 3 - IDE for Python 3](https://play.google.com/store/apps/details/Pydroid_3_IDE_for_Python_3).

2. **Install the Pydroid Repository Plugin:**
   - To enhance functionality, install the Pydroid repository plugin. Find it on the Google Play Store or use this link: [Pydroid Repository Plugin](https://play.google.com/store/apps/details?id=ru.iiec.pydroid3.quickinstallrepo).

3. **Adjust App Settings:**
   - In the app settings for Pydroid, disable power-saving mode and ensure that the option to pause when not in use is also disabled. This ensures uninterrupted operation of your Python scripts.

4. **Install Required Packages:**
   - Open Pip within the Pydroid app and install these necessary packages:
     ```
      g4f flask pillow beautifulsoup4
     ```

5. **Create a New Python Script:**
   - Within Pydroid, create a new Python script and input the following content:
     ```python
     from g4f import set_cookies

     set_cookies(".bing.com", {
         "_U": "cookie value"
     })

     from g4f.gui import run_gui

     run_gui("0.0.0.0", 8080, debug=True)
     ```
     Replace `"cookie value"` with your actual cookie value from Bing if you intend to create images using Bing.

6. **Execute the Script:**
   - Run the script by clicking on the play button or selecting the option to execute it.

7. **Access the GUI:**
   - Wait for the server to start, and once it's running, open the GUI using the URL provided in the output. [http://localhost:8080/chat/](http://localhost:8080/chat/)

By following these steps, you can successfully run the G4F GUI on your smartphone using Pydroid, allowing you to create and interact with graphical interfaces directly from your device.