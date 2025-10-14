import cv2  # OpenCV library for video processing and display
import numpy as np  # NumPy library for numerical operations and handling arrays
from datetime import datetime  # For timestamping CSV entries
import os  # For file path manipulations and directory operations
import csv  # For reading and writing CSV files

# Global variables to keep track of the current frame, total frames, and playback state
current_frame_idx = 0
total_frames = 0
playing = True
zoom_factor = 1.0  # For zoom in/out functionality

# Define the number of frames to skip when navigating forward or backward
FRAME_SKIP = 10  # Adjust this value based on desired skip size
SLOW_SKIP = 1 #This value is used to skip one frame 


def enhance_frame(frame,colormap):
    """
    Enhances the input frame by normalizing its pixel values and applying a colormap for better visibility.

    Parameters:
    - frame (numpy.ndarray): The original grayscale or single-channel frame.

    Returns:
    - numpy.ndarray: The enhanced frame with applied colormap.
    """
    # Normalize pixel values to range between 0 and 255 for better contrast
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply the JET colormap to the normalized frame
    enhanced_frame = cv2.applyColorMap(frame_normalized.astype(np.uint8), colormap)
    
    return enhanced_frame

# Retinex filter function from plume_visualizer
def apply_retinex_filter(frame, radii, eps, weights, gf_scales):
    """
    Apply Retinex filter to enhance gas visibility in IR frames.
    Uses guided filter scales and parameters.
    """
    # Retinex logic, filtering the frame with guided filters (as an example)
    processed_frame = cv2.ximgproc.guidedFilter(frame, frame, 5, eps[0])
    # Apply weightings, gain, bias, and normalization (simplified for integration)
    processed_frame = cv2.normalize(processed_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return processed_frame


# Switch colormap based on user input (key pressed)
def switch_colormap(key):
    if key == ord('1'): 
        return cv2.COLORMAP_JET
    elif key == ord('2'):
        return cv2.COLORMAP_BONE  # Placeholder for GrayRain50
    elif key == ord('3'):
        return cv2.COLORMAP_RAINBOW
    elif key == ord('4'):
        return cv2.COLORMAP_OCEAN  # Placeholder for Research
    else:
        return None

def load_npy_video(file_path):
    """
    Loads video frames from a .npy file. The .npy file is expected to contain a 3D NumPy array
    representing the video frames.

    Parameters:
    - file_path (str): Path to the .npy video file.

    Returns:
    - numpy.ndarray: Array of video frames.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - ValueError: If the loaded array does not have the expected dimensions.
    """
    # Check if the specified file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    try:
        # Load the video frames from the .npy file
        video = np.load(file_path)
        
        # Validate that the loaded array has 3 dimensions (frames, height, width)
        if video.ndim != 3:
            raise ValueError("Expected video frames to have 3 dimensions (frames, height, width).")
        
        return video
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        raise

def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse click events to allow users to seek to a specific frame by clicking on the progress bar.

    Parameters:
    - event: The type of mouse event (e.g., button down, button up).
    - x (int): The x-coordinate of the mouse event.
    - y (int): The y-coordinate of the mouse event.
    - flags: Any relevant flags passed by OpenCV.
    - param (dict): A dictionary containing additional parameters, specifically the progress bar dimensions.
    """
    global current_frame_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        # Retrieve the progress bar dimensions from the parameters
        progress_bar = param.get('progress_bar')
        if progress_bar:
            bar_x, bar_y, bar_width, bar_height = progress_bar
            
            # Check if the mouse click occurred within the bounds of the progress bar
            if bar_x <= x <= bar_x + bar_width and bar_y <= y <= bar_y + bar_height:
                # Calculate the ratio of the click position relative to the progress bar width
                clicked_ratio = (x - bar_x) / bar_width
                
                # Clamp the ratio between 0 and 1 to avoid invalid indices
                clicked_ratio = max(0, min(clicked_ratio, 1))
                
                # Determine the new frame index based on the clicked position
                new_frame_idx = int(clicked_ratio * total_frames)
                current_frame_idx = new_frame_idx
                print(f'Frame jumped to: {current_frame_idx}')

def draw_progress_bar(frame, current_frame_idx, total_frames):
    """
    Draws a progress bar on the video frame to indicate the current playback position.

    Parameters:
    - frame (numpy.ndarray): The video frame on which to draw the progress bar.
    - current_frame_idx (int): The index of the current frame being displayed.
    - total_frames (int): The total number of frames in the video.

    Returns:
    - tuple: A tuple containing the progress bar's x and y coordinates, width, and height.
    """
    # Define the height of the progress bar and padding from the frame edges
    bar_height = 20
    padding = 10
    
    # Retrieve the frame's dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Dynamically calculate the width of the progress bar based on the frame's width
    bar_width = frame_width - 2 * padding
    bar_x = padding
    bar_y = frame_height - bar_height - padding  # Position the bar near the bottom of the frame

    # Draw the background of the progress bar (white)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), -1)
    
    # Calculate the width of the filled portion of the bar (green) based on playback progress
    filled_width = int(bar_width * (current_frame_idx / total_frames))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
    
    return (bar_x, bar_y, bar_width, bar_height)

def save_entry_to_csv(csv_file, entries):
    """
    Saves frame marking entries to a CSV file. If the directory does not exist, it is created.
qq
    Parameters:
    - csv_file (str): Path to the CSV file where entries will be saved.
    - entries (list of lists): A list of entries, where each entry is a list containing details about frame segments.
    """
    # Create the directory for the CSV file if it doesn't already exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    try:
        # Open the CSV file in write mode
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header row
            writer.writerow(['Type', 'Start Frame', 'End Frame', 'Duration (Frames)', 'Duration (Seconds)', 'Last Updated'])
            
            # Write all entries to the CSV
            writer.writerows(entries)
    except Exception as e:
        print(f"Error saving CSV file: {e}")

def zoom_frame(frame, zoom_factor):
    """
    Zooms in or out on the frame based on the zoom factor.

    Parameters:
    - frame (numpy.ndarray): The video frame to be zoomed.
    - zoom_factor (float): The zoom factor (e.g., 1.0 for no zoom, 2.0 for 2x zoom).

    Returns:
    - numpy.ndarray: The zoomed frame.
    """
    if zoom_factor == 1.0:
        return frame

    h, w = frame.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    zoomed_frame = frame[top:top + new_h, left:left + new_w]
    return cv2.resize(zoomed_frame, (w, h), interpolation=cv2.INTER_LINEAR) 

# Integrated video processing function
def process_video_with_retinex(video_path):
    """
    Process video frames and apply the Retinex filter for better gas leak visibility.
    """
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Apply Retinex filter to enhance frame visibility
            radii = [15, 80]  # Example values for retinex parameters
            eps = [0.3, 0.4]  # Example values for guided filter epsilons
            weights = [0.6, 0.4]  # Example weightings
            gf_scales = [5, 7]  # Guided filter scales
            frame_retinex = apply_retinex_filter(frame, radii, eps, weights, gf_scales)
            
            # Further processing can be added here (e.g., saving data to CSV)
            # save_to_csv(frame_retinex)

            # Display the Retinex-enhanced frame
            cv2.imshow('Retinex-enhanced Frame', frame_retinex)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def main(video_file, csv_folder):
    """
    The main function that orchestrates video playback, user interactions, and CSV logging.

    Parameters:
    - video_file (str): Path to the .npy video file to be played.
    - csv_folder (str): Directory where the CSV log file will be saved.
    """
    global current_frame_idx, total_frames, playing ,zoom_factor
   
    
    # Display the NumPy version being used
    print(f"Using NumPy version: {np.__version__}")
    
    # Load the video frames from the specified .npy file
    video_frames = load_npy_video(video_file)
    total_frames = video_frames.shape[0]
    print(f"Total frames loaded: {total_frames}")
    
    # Initialize playback parameters
    colormap = cv2.COLORMAP_JET ##A
    current_frame_idx = 0
    start_frame_idx = None  # To mark the start of an event (e.g., gas leak)
    end_frame_idx = None    # To mark the end of an event
    fps = 30  # Frames per second; adjust if your video has a different frame rate
    playing = True  # Flag to control whether the video is playing or paused
    
    # Extract the base name of the video file (without extension) for naming the CSV
    original_video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    # Create a window named 'Video' to display the frames
    cv2.namedWindow('Video',cv2.WINDOW_NORMAL) # Allow window resizing
    
    # Dictionary to store progress bar dimensions, passed to the mouse callback
    progress_bar_params = {'progress_bar': None}
    
    # Set the mouse callback function for the 'Video' window
    cv2.setMouseCallback('Video', mouse_callback, param=progress_bar_params)
    
    while True:
        # Retrieve the current frame to be displayed
        frame = video_frames[current_frame_idx]
        
        # Enhance the frame for better visibility
        enhanced_frame = enhance_frame(frame,colormap)
        
        # Create a copy of the enhanced frame to overlay text and graphics
        frame_with_text = enhanced_frame.copy()

         # Allow window resizing
        zoomed_frame = zoom_frame(enhanced_frame, zoom_factor)  # Apply the zoom to the enhanced frame
        frame_with_text = zoomed_frame.copy()  # Now this will work since zoomed_frame is defined
                                               # and also it createss the copy of the both enhanced frame and Zommed frame  

        
        # Overlay the current frame number on the top-left corner
        cv2.putText(
            frame_with_text, 
            f'Frame: {current_frame_idx}', 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Overlay instructions for user interactions below the frame number
        instructions = "'q' to Quit, 's' to Mark Frame, 'p' to Pause/Play, 'r' to Reset, 'f' Forward, 'b' Backward"
        cv2.putText(
            frame_with_text, 
            instructions, 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            1
        )
        
        # Draw the progress bar and update its parameters
        progress_bar = draw_progress_bar(frame_with_text, current_frame_idx, total_frames)
        progress_bar_params['progress_bar'] = progress_bar

        # Display the frame with overlays in the 'Video' window
        cv2.imshow('Video', frame_with_text)
        
        # Check if the current frame is the last frame of the video
        if current_frame_idx >= total_frames - 1:
            print(f'Last frame number: {current_frame_idx}')
            print('Reached the last frame. Waiting for user input before exiting.')
            # Wait indefinitely for a key press before exiting
            key = cv2.waitKey(0) & 0xFF
        else:
            # Wait for 30 milliseconds and capture any key press
            key = cv2.waitKey(30) & 0xFF
        
        # Handle user key presses
        if key == ord('q'):
            # 'q' key to quit the video playback
            print('Quitting the video playback.')
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:  # Switch colormap
            new_colormap = switch_colormap(key)
            if new_colormap:
                colormap = new_colormap
                print(f"Switched colormap to option {chr(key)}")
        elif key == ord('s'):
            # 's' key to mark the start or end frame of an event
            if start_frame_idx is None:
                start_frame_idx = current_frame_idx
                print(f'Start frame marked at: {start_frame_idx}')
            elif end_frame_idx is None:
                end_frame_idx = current_frame_idx
                print(f'End frame marked at: {end_frame_idx}')
        elif key == ord('p'):
            # 'p' key to toggle between pausing and playing the video
            playing = not playing
            print('Video paused' if not playing else 'Video playing')
        elif key == ord('r'):
            # 'r' key to reset the video playback to the beginning
            current_frame_idx = 0
            print('Video reset to the beginning')
        elif key == ord('f'):
            # 'f' key to skip forward by FRAME_SKIP frames
            current_frame_idx = min(current_frame_idx + FRAME_SKIP, total_frames - 1)
            print(f'Moved forward to frame: {current_frame_idx}')
        elif key == ord('b'):
            # 'b' key to skip backward by FRAME_SKIP frames
            current_frame_idx = max(current_frame_idx - FRAME_SKIP, 0)
            print(f'Moved backward to frame: {current_frame_idx}')

        elif key == ord('l'):
            # 'f' key to skip forward by FRAME_SKIP frames
            current_frame_idx = min(current_frame_idx + SLOW_SKIP, total_frames - 1)
            print(f'Moved forward to frame: {current_frame_idx}')
        elif key == ord('k'):
            # 'b' key to skip backward by FRAME_SKIP frames
            current_frame_idx = max(current_frame_idx - SLOW_SKIP, 0)
            print(f'Moved backward to frame: {current_frame_idx}')
        elif key == ord('+'):
            zoom_factor = min(zoom_factor + 0.1, 3.0)
            print(f"Zooming in: {zoom_factor}x")
        elif key == ord('-'):
            zoom_factor = max(zoom_factor - 0.1, 1.0)
            print(f"Zooming out: {zoom_factor}x")
        
        # If the video is in playing state, increment the frame index
        if playing:
            current_frame_idx += 1
            # Ensure the frame index does not exceed the total number of frames
            if current_frame_idx >= total_frames:
                current_frame_idx = total_frames - 1  # Stay on the last frame
    
    # After exiting the playback loop, prepare entries for the CSV log
    entries = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp
    
    if start_frame_idx is not None and end_frame_idx is not None:
        # If both start and end frames are marked, log different segments
        
        # 1. Non-Leaking Part 1: from frame 0 to start_frame_idx - 1
        if start_frame_idx > 0:
            entries.append([
                'null',                           # Type
                0,                                # Start Frame
                start_frame_idx - 1,              # End Frame
                start_frame_idx,                  # Duration (Frames)
                start_frame_idx / fps,            # Duration (Seconds)
                current_time                      # Last Updated
            ])
        
        # 2. Leaking Part: from start_frame_idx to end_frame_idx
        entries.append([
            'gas',                              # Type
            start_frame_idx,                    # Start Frame
            end_frame_idx,                      # End Frame
            end_frame_idx - start_frame_idx,    # Duration (Frames)
            (end_frame_idx - start_frame_idx) / fps,  # Duration (Seconds)
            current_time                        # Last Updated
        ])
        
        # 3. Non-Leaking Part 2: from end_frame_idx + 1 to total_frames - 1
        if end_frame_idx < total_frames - 1:
            entries.append([
                'null',                           # Type
                end_frame_idx + 1,                # Start Frame
                total_frames - 1,                  # End Frame
                total_frames - end_frame_idx - 1,  # Duration (Frames)
                (total_frames - end_frame_idx - 1) / fps,  # Duration (Seconds)
                current_time                        # Last Updated
            ])
    else:
        # If no frames are marked, log the entire video as 'null'
        entries.append([
            'null',                               # Type
            0,                                    # Start Frame
            total_frames - 1,                      # End Frame
            total_frames,                          # Duration (Frames)
            total_frames / fps,                    # Duration (Seconds)
            current_time                            # Last Updated
        ])
    
    # Define the path for the CSV log file, naming it after the original video
    csv_file = os.path.join(csv_folder, f"{original_video_name}.csv")
    
    # Close all OpenCV windows before prompting the user
    cv2.destroyAllWindows()
    
    # new Save the prepared entries to the CSV file
    if entries:
        # Print the header and the data that will be saved to the CSV
        header = ['Type', 'Start Frame', 'End Frame', 'Duration (Frames)', 'Duration (Seconds)', 'Last Updated']
        print("Entries to be saved in CSV:")
        print(header)  # Print header row
        for entry in entries:
            print(entry)  # Print each entry being saved

       #save_entry_to_csv(csv_file, entries)
        #print(f"CSV '{csv_file}' was last updated at: {current_time}")

    # Prompt the user to confirm whether to update the CSV file
    while True:
        user_input = input("Do you want to update the CSV file? (Y/N): ").strip().upper()
        if user_input == 'Y':
            # If user confirms, save the entries to the CSV file
            if entries:
                save_entry_to_csv(csv_file, entries)
                print(f"CSV '{csv_file}' was last updated at: {current_time}")
            break
        elif user_input == 'N':
            # If user declines, skip saving the CSV file
            print("CSV update skipped.")
            break
        else:
            # If input is invalid, prompt again
            print("Invalid input. Please enter 'Y' or 'N'.")

# Entry point of the script
if __name__ == "__main__":
    # Specify the path to the input video file (in .npy format)
    video_file = r'C:\Users\Ajay\Desktop\npy\videoafternuc (5).npy'  # Specify your video path

    # Specify the directory where the CSV output will be saved
    csv_folder = r'C:\Users\Ajay\Desktop\Output'  # Specify your desired CSV folder path

    # Call the function to process video with Retinex filter
    process_video_with_retinex(video_file)

    # Call the main function with the specified video file and CSV folder
    main(video_file, csv_folder)

