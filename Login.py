# Displays a login page for user authentication.
# Uses SQLite database to store user credentials.
# Validates login and redirects to Output.py.

# Import necessary libraries
import streamlit as st  # Streamlit is used to create a web-based application
import base64  # Base64 is used for encoding image files to display as background images
import cv2  # OpenCV library for image processing
import sqlite3  # SQLite3 is used to manage the user database

# ====================== Display Application Title ======================

# Displaying the main title for the application at the top of the web page
st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"CO2WOUNDS-V2 extended chronic wounds dataset from leprosy patients"}</h1>', unsafe_allow_html=True)

# ====================== Function to Set Background Image ======================
def add_bg_from_local(image_file):
    """Function to add a background image from a local file."""
    with open(image_file, "rb") as image_file:  # Open image file in binary mode
        encoded_string = base64.b64encode(image_file.read())  # Encode image file as base64 string
    # Set the background image using custom CSS styling
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});  /* Convert encoded image back to display */
        background-size: cover  /* Ensure the background image covers the full screen */
    }}
    </style>
    """,
    unsafe_allow_html=True  # Allow raw HTML/CSS in Streamlit
    )
    
# Calling function to set background image
add_bg_from_local('2.jpg')

# ====================== Display Login Header ======================
st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;font-family:Caveat, sans-serif;">{"Login Here !!!"}</h1>', unsafe_allow_html=True)

# ====================== Function to Create Database Connection ======================
def create_connection(db_file):
    """Establish a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)  # Connect to SQLite database
    except sqlite3.Error as e:
        print(e)  # Print error message if connection fails
    return conn  # Return the database connection object

# ====================== Function to Insert New User into Database ======================
def create_user(conn, user):
    """Insert a new user into the 'users' table."""
    sql = ''' INSERT INTO users(name, password, email, phone)  # SQL query to insert user data
              VALUES(?,?,?,?) '''  # Use placeholders for user data
    cur = conn.cursor()  # Create a cursor object to execute queries
    cur.execute(sql, user)  # Execute query with provided user data
    conn.commit()  # Save changes to the database
    return cur.lastrowid  # Return the ID of the newly inserted user

# ====================== Function to Validate User Login Credentials ======================
def validate_user(conn, name, password):
    """Validate user login credentials against the database."""
    cur = conn.cursor()  # Create a cursor object to execute queries
    cur.execute("SELECT * FROM users WHERE name=? AND password=?", (name, password))  # Query to check user credentials
    user = cur.fetchone()  # Fetch the first matching record
    if user:
        return True, user[1]  # If user exists, return True and username
    return False, None  # Otherwise, return False and None

# ====================== Main Function to Handle User Login ======================
def main():
    """Main function to handle user login process."""
    
    # Establish connection to the database
    conn = create_connection("dbs.db")

    if conn is not None:  # Check if the connection was successful
        # Create the 'users' table if it does not already exist
        conn.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY,  /* Auto-incrementing user ID */
                     name TEXT NOT NULL,  /* Username (Required) */
                     password TEXT NOT NULL,  /* Password (Required) */
                     email TEXT NOT NULL UNIQUE,  /* Unique Email (Required) */
                     phone TEXT NOT NULL);''')  /* Phone number (Required) */

        # Display input fields for user login
        st.write("Enter your credentials to login:")
        name = st.text_input("User name")  # Input field for username
        password = st.text_input("Password", type="password")  # Input field for password (masked)

        # Creating columns for button placement
        col1, col2 = st.columns(2)

        with col1:
            # Login Button
            login_button = st.button("Login")  # Button to trigger login process
            
            if login_button:  # If login button is clicked
                is_valid, user_name = validate_user(conn, name, password)  # Validate user credentials
                
                if is_valid:  # If credentials are correct
                    st.success(f"Welcome back, {user_name}! Login successful!")  # Display success message
                    
                    # Run Output.py script using subprocess to proceed to next step
                    import subprocess
                    subprocess.run(['python', '-m', 'streamlit', 'run', 'Output.py'])

                else:  # If credentials are incorrect
                    st.error("Invalid user name or password!")  # Display error message

        # Close the database connection after user login attempt
        conn.close()
    else:
        st.error("Error! cannot create the database connection.")  # Display error if database connection fails

# ====================== Run the Main Function ======================
if __name__ == '__main__':
    main()

    
        
        
        
        
        
        
        
        
        
        
        
        