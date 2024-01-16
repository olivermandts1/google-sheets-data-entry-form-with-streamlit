import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.subheader("📗 Refresh FB Creatives From Your Plutus Spreadsheet")

st.write("#### 1. Retrieved Plutus Creatives")

# Create a connection using Streamlit's experimental connection feature
conn = st.experimental_connection("gsheets", type=GSheetsConnection)

# Read data from the Google Sheet
df = conn.read(worksheet="PlutusDataImport", ttl=10)
desired_range = df.iloc[99:124, 0:2]  # Rows 100-124 and columns A-B (0-indexed)

# Hardcoding specific values in column A
desired_range.iloc[0:5, 0] = 'Headlines'      # Rows 100-104
desired_range.iloc[6:11, 0] = 'Primary Text'  # Rows 106-110
desired_range.iloc[12:17, 0] = 'Description'  # Rows 112-116
desired_range.iloc[19:24, 0] = 'Forcekeys'    # Rows 119-123

# Replace NaN values with an empty string
desired_range.fillna('', inplace=True)

# Rename the columns
desired_range.columns = ['Asset Type', 'Creative Text']

# Store values, omitting nulls
headlines = [text for text in desired_range[desired_range['Asset Type'] == 'Headlines']['Creative Text'] if text]
primary_text = [text for text in desired_range[desired_range['Asset Type'] == 'Primary Text']['Creative Text'] if text]
descriptions = [text for text in desired_range[desired_range['Asset Type'] == 'Description']['Creative Text'] if text]
forcekeys = [text for text in desired_range[desired_range['Asset Type'] == 'Forcekeys']['Creative Text'] if text]

# Use st.expander to create a toggle for showing the full table
with st.expander("Show Full Table", expanded=True):
    # Use st.markdown with HTML and CSS to enable text wrapping inside the expander
    st.markdown("""
    <style>
    .dataframe th, .dataframe td {
        white-space: nowrap;
        text-align: left;
        border: 1px solid black;
        padding: 5px;
    }
    .dataframe th {
        background-color: #f0f0f0;
    }
    .dataframe td {
        min-width: 50px;
        max-width: 700px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: normal;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display the DataFrame with text wrapping inside the expander
    st.markdown(desired_range.to_html(escape=False, index=False), unsafe_allow_html=True)

# Display Title and Description
st.title("Vendor Management Portal")

# Constants
BUSINESS_TYPES = [
    "Manufacturer",
    "Distributor",
    "Wholesaler",
    "Retailer",
    "Service Provider",
]
PRODUCTS = [
    "Electronics",
    "Apparel",
    "Groceries",
    "Software",
    "Other",
]

# Establishing a Google Sheets connection
conn = st.experimental_connection("gsheets", type=GSheetsConnection)

# Fetch existing vendors data
existing_data = conn.read(worksheet="Test", usecols=list(range(6)), ttl=5)
existing_data = existing_data.dropna(how="all")

action = st.selectbox(
    "Choose an Action",
    [
        "Onboard New Vendor",
        "Update Existing Vendor",
        "View All Vendors",
        "Delete Vendor",
    ],
)

if action == "Onboard New Vendor":
    st.markdown("Enter the details of the new vendor below.")
    with st.form(key="vendor_form"):
        company_name = st.text_input(label="Company Name*")
        business_type = st.selectbox(
            "Business Type*", options=BUSINESS_TYPES, index=None
        )
        products = st.multiselect("Products Offered", options=PRODUCTS)
        years_in_business = st.slider("Years in Business", 0, 50, 5)
        onboarding_date = st.date_input(label="Onboarding Date")
        additional_info = st.text_area(label="Additional Notes")

        st.markdown("**required*")
        submit_button = st.form_submit_button(label="Submit Vendor Details")

        if submit_button:
            if not company_name or not business_type:
                st.warning("Ensure all mandatory fields are filled.")
            elif existing_data["CompanyName"].str.contains(company_name).any():
                st.warning("A vendor with this company name already exists.")
            else:
                vendor_data = pd.DataFrame(
                    [
                        {
                            "CompanyName": company_name,
                            "BusinessType": business_type,
                            "Products": ", ".join(products),
                            "YearsInBusiness": years_in_business,
                            "OnboardingDate": onboarding_date.strftime("%Y-%m-%d"),
                            "AdditionalInfo": additional_info,
                        }
                    ]
                )
                updated_df = pd.concat([existing_data, vendor_data], ignore_index=True)
                conn.update(worksheet="Test", data=updated_df)
                st.success("Vendor details successfully submitted!")

elif action == "Update Existing Vendor":
    st.markdown("Select a vendor and update their details.")

    vendor_to_update = st.selectbox(
        "Select a Vendor to Update", options=existing_data["CompanyName"].tolist()
    )
    vendor_data = existing_data[existing_data["CompanyName"] == vendor_to_update].iloc[
        0
    ]

    with st.form(key="update_form"):
        company_name = st.text_input(
            label="Company Name*", value=vendor_data["CompanyName"]
        )
        business_type = st.selectbox(
            "Business Type*",
            options=BUSINESS_TYPES,
            index=BUSINESS_TYPES.index(vendor_data["BusinessType"]),
        )
        products = st.multiselect(
            "Products Offered",
            options=PRODUCTS,
            default=vendor_data["Products"].split(", "),
        )
        years_in_business = st.slider(
            "Years in Business", 0, 50, int(vendor_data["YearsInBusiness"])
        )
        onboarding_date = st.date_input(
            label="Onboarding Date", value=pd.to_datetime(vendor_data["OnboardingDate"])
        )
        additional_info = st.text_area(
            label="Additional Notes", value=vendor_data["AdditionalInfo"]
        )

        st.markdown("**required*")
        update_button = st.form_submit_button(label="Update Vendor Details")

        if update_button:
            if not company_name or not business_type:
                st.warning("Ensure all mandatory fields are filled.")
            else:
                # Removing old entry
                existing_data.drop(
                    existing_data[
                        existing_data["CompanyName"] == vendor_to_update
                    ].index,
                    inplace=True,
                )
                # Creating updated data entry
                updated_vendor_data = pd.DataFrame(
                    [
                        {
                            "CompanyName": company_name,
                            "BusinessType": business_type,
                            "Products": ", ".join(products),
                            "YearsInBusiness": years_in_business,
                            "OnboardingDate": onboarding_date.strftime("%Y-%m-%d"),
                            "AdditionalInfo": additional_info,
                        }
                    ]
                )
                # Adding updated data to the dataframe
                updated_df = pd.concat(
                    [existing_data, updated_vendor_data], ignore_index=True
                )
                conn.update(worksheet="Test", data=updated_df)
                st.success("Vendor details successfully updated!")

# View All Vendors
elif action == "View All Vendors":
    st.dataframe(existing_data)

# Delete Vendor
elif action == "Delete Vendor":
    vendor_to_delete = st.selectbox(
        "Select a Vendor to Delete", options=existing_data["CompanyName"].tolist()
    )

    if st.button("Delete"):
        existing_data.drop(
            existing_data[existing_data["CompanyName"] == vendor_to_delete].index,
            inplace=True,
        )
        conn.update(worksheet="Test", data=existing_data)
        st.success("Vendor successfully deleted!")
