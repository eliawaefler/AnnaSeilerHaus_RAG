# app to create IDS

import streamlit as st
import ifcopenshell
import bcrypt
import psycopg2
import os


def setup_pw_db():
    # Get database credentials securely
    db_user = st.secrets["db_username"]
    db_password = st.secrets["db_password"]
    db_host = st.secrets["db_host"]
    db_name = st.secrets["db_name"]

    # Connect to your database
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host
    )

    # Hash a password for storing.
    def hash_password(password):
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt)

    def check_password(stored_password, provided_password):
        return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

    # Example of adding a new user (simplified)
    def add_user(username, password):
        cursor = conn.cursor()
        hashed_password = hash_password(password)
        query = "INSERT INTO users (username, password) VALUES (%s, %s)"
        cursor.execute(query, (username, hashed_password))
        conn.commit()
        cursor.close()


def simplify_ifc(input_file):
    # Load the original IFC file
    original_ifc = ifcopenshell.open(input_file)

    # Dictionary to store one example of each type
    unique_elements = {}

    # Iterate over all elements in the IFC file
    for element in original_ifc.by_type('IfcProduct'):
        element_type = element.is_a()
        # Store the first occurrence of each type
        if element_type not in unique_elements:
            unique_elements[element_type] = element

    # Create a new IFC file to store the simplified model
    schema = original_ifc.schema
    new_ifc_file = ifcopenshell.file(schema=schema)

    # Copy elements to the new IFC file, maintaining all associated information
    for element in unique_elements.values():
        # Copy element and its properties
        new_element = new_ifc_file.add(element)
        # Recursively copy all related objects

        def copy_related_objects(obj):
            for related_obj in obj:
                copied_obj = new_ifc_file.add(related_obj)
                copy_related_objects(copied_obj)

        copy_related_objects(new_element)

    # Optionally, save the new IFC file
    output_file = 'simplified.ifc'
    new_ifc_file.write(output_file)

    return output_file


def set_username():
    st.session_state.user_pw = True


def display_home():
    st.title("welcome to the IDS creator")

    if st.button("sign in"):
        st.session_state.page = "sign_in"
        return 0
    if st.button("sing up"):
        st.session_state.page = "sign_up"
        return 0
    if st.button("create new IDS"):
        st.session_state.page = "create_ids"
        return 0


def display_sing_up():
    st.title("sing up")
    st.session_state.username = st.text_input("email")
    st.session_state.username = st.text_input("username")
    st.session_state.user_pw = st.text_input("password", type="password", on_change=set_username)
    if st.button("back"):
        st.session_state.page = "home"
        return 0


def display_sing_in():
    st.title("sign in")
    st.title(f"Welcome {st.session_state.username}")
    if st.session_state.user_pw:
        st.session_state.page = "user_page"
    st.session_state.username = st.text_input("username")
    st.session_state.user_pw = st.text_input("password", type="password", on_change=set_username)
    if st.button("forgot password"):
        st.session_state.page = "forgot_pw"
        return 0
    if st.button("back"):
        st.session_state.page = "home"
        return 0


def display_forgot_pw():
    st.title("forgot pw")
    st.text_input("email or username")
    if st.button("send link"):
        st.session_state.page = "home"
        return 0
    if st.button("back"):
        st.session_state.page = "sign_in"
        return 0


def display_create_ids():
    st.title("IDS creator")
    col1, col2, col3 = st.columns([3, 1, 3])

    with col1:
        st.write("if you want to create IDS from an IFC, "
                 "or if you have an existing IDS, upload the file here")
        uploaded_file = st.file_uploader("upload IFC or IDS", accept_multiple_files=False)

        # Create a directory for uploaded files if it doesn't exist
        upload_folder = "uploaded_files"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        if st.button("process"):
            if uploaded_file:

                # safe uploaded file
                file_path = os.path.join(upload_folder, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"File '{uploaded_file.name}' has been uploaded to '{file_path}'")

                if ".ifc" in uploaded_file.name:
                    st.session_state.file = uploaded_file.name
                    try:
                        simple_ifc = simplify_ifc(file_path)
                        st.success("IFC ingested")
                        st.button("download simplified IFC")

                        # safe simplified ifc
                        file_path = os.path.join(upload_folder, simple_ifc)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.success(f"File '{simple_ifc}' has been uploaded to '{file_path}'")

                    except:
                        st.error("IFC error")
                else:
                    st.session_state.file = "ids" + str(uploaded_file.name)
                    st.success("IDS ingested")

            else:
                st.warning("no files uploaded")

    with col3:
        st.write("else you can choose your use cases here:")
        st.toggle("use case 1")
        st.toggle("use case 2")
        st.toggle("use case 3")

        if st.button("create new use case"):
            st.session_state.page = "create_use_case"
            return 0

    st.write("")
    st.write("")

    if "ids" in st.session_state.file:
        if st.button("show IDS Use Cases"):
            st.write("the uploaded file supports Use cases 1 and 2, but not 3")  # sample result
    elif "ifc" in st.session_state.file:
        if st.button("show IFC Use Cases"):
            st.write("the uploaded file supports Use cases 1 and 2, but not 3")  # sample result

    ids_name = st.text_input("new IDS name")
    st.download_button(label="create and download IDS", data="hello world",   # add funcitonality
                       file_name=ids_name, mime="text/plain")

    if st.button("back"):
        st.session_state.page = "home"
        return 0


def display_create_use_case():
    st.title("create use case")
    if st.button("back"):
        st.session_state.page = "home"
        return 0


def main():
    # initialize session states
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "user_pw" not in st.session_state:
        st.session_state.user_pw = False
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "file" not in st.session_state:
        st.session_state.file = ""

    # Define the navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("Home", key="nav_home"):
        st.session_state.page = "home"
    if st.sidebar.button("sing in", key="nav_sing_in"):
        st.session_state.page = "sign_in"
    if st.sidebar.button("sing up", key="nav_sing_up"):
        st.session_state.page = "sign_up"
    if st.sidebar.button("create IDS", key="nav_create_ids"):
        st.session_state.page = "create_ids"
    if st.sidebar.button("create UseCase", key="nav_create_use_case"):
        st.session_state.page = "create_use_case"
    if st.sidebar.button("About", key="nav_about"):
        st.session_state.page = "about"
    if st.sidebar.button("Contact", key="nav_contact"):
        st.session_state.page = "contact"

    # Page functionality
    if st.session_state.page == "home":
        display_home()
    elif st.session_state.page == "sign_in":
        display_sing_in()
    elif st.session_state.page == "sign_up":
        display_sing_up()
    elif st.session_state.page == "forgot_pw":
        display_forgot_pw()
    elif st.session_state.page == "create_ids":
        display_create_ids()
    elif st.session_state.page == "create_use_case":
        display_create_use_case()
    elif st.session_state.page == "about":
        st.title("About Page")
        st.write("Welcome to the About Page!")
    elif st.session_state.page == "contact":
        st.title("Contact Page")
        st.write("Welcome to the Contact Page! Please reach out to us.")


if __name__ == '__main__':
    main()
