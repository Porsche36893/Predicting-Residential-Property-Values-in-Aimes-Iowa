#Importing Library

#Modeling import
import joblib
# Data Analysis
import numpy as np
import pandas as pd
from scipy.stats import boxcox

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Data Modeling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from xgboost import XGBRegressor


#Streaming
import streamlit as st


#Page editing
st.set_page_config(
    page_title="Residential Property Prediction Tools!",
    layout="wide"  # Use the 'wide' layout
)
#input frame
req_full = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',
       'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
       'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'SaleType', 'SaleCondition']
input = pd.DataFrame(columns= req_full)

# Custom HTML and CSS to style the text
st.markdown(
    """
    <style>
    .custom-header {
        font-size: 50px; /* Adjust size as needed */
        font-weight: bold;
    }
    .gold {
        color: gold; /* Gold color for "I" */
    }
    .cardinal {
        color: #8C1515; /* Cardinal color for the rest */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Render the styled text

st.image("logo2.png", use_container_width= True)

# Title text
st.write("# Greetings Ames, Iowa citizens! ")

st.write("##### _Welcome to our House Price Prediction tool, designed specifically for the vibrant community of Ames, Iowa. Whether you're a seasoned homeowner looking to assess your property's value or a prospective buyer exploring the market, our accurate and user-friendly tool is here to assist you. Our model leverages advanced machine learning techniques and a comprehensive dataset of historical home sales in Ames to provide reliable price estimates. By inputting key property details, you can gain valuable insights into potential market values and make informed decisions._")
st.text("")
st.markdown(
    """
    <style>
    .gold-banner {
        width: 100%; /* Full width of the page */
        height: 20px; /* Height of the banner */
        background-color: #eccd65; /* Gold color */
        text-align: center; /* Center-align text inside the banner */
        line-height: 75px; /* Vertically center text */
        font-size: 24px; /* Text size */
        color: black; /* Text color */
        font-weight: bold; /* Bold text */
    }
    </style>
    <div class="gold-banner"></div>
    """,
    unsafe_allow_html=True
)
st.text("")

col1, col2, col3 = st.columns([1, 0.1, 1])

with col1:
    st.header("Did you know? ")
    st.write("Ames, Iowa, is home to a unique architectural style known as the Prairie School. This style, popularized by architects like Frank Lloyd Wright, emphasizes horizontal lines, open floor plans, and a strong connection to the natural environment. Many historic homes in Ames showcase this distinctive architectural heritage, making it a fascinating place for architecture enthusiasts to explore.")
    st.image("prairie_bobshimer.jpg", caption="Prairie style buildings", use_container_width=True)
with col2:
    st.write(" ")
with col3:
    st.header("Why our website?")
    st.write("Knowing your home's value is crucial for Ames residents, whether you're planning to sell, refinance, or simply understand your financial situation. Our model empowers you with accurate and accessible home price estimates, challenging traditional real estate practices. We offer a competitive alternative to large real estate companies, providing you with the information you need to make informed decisions about your property.")
    st.image("Real-Estate-3d-2.jpg", caption="Prairie style buildings", use_container_width=True)

st.text("")
st.markdown(
    """
    <style>
    .gold-banner {
        width: 100%; /* Full width of the page */
        height: 20px; /* Height of the banner */
        background-color: #eccd65; /* Gold color */
        text-align: center; /* Center-align text inside the banner */
        line-height: 75px; /* Vertically center text */
        font-size: 24px; /* Text size */
        color: black; /* Text color */
        font-weight: bold; /* Bold text */
    }
    </style>
    <div class="gold-banner"></div>
    """,
    unsafe_allow_html=True
)
st.text("")
st.header("Prediction Terminal")
st.text("Note: Before adding any value, Please consult and take a look at attribues description list for each attributes that you would input into our model. ")

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Only input numerical values")
    # Numerical Columns
    input.loc[0, "MSSubClass"] = st.selectbox("MSSubClass: The building class",([None,20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190]))
    input.loc[0, "LotFrontage"] = st.number_input("LotFrontage: Linear feet of street connected to property")
    input.loc[0, "LotArea"] = st.number_input("LotArea: Lot size in square feet")
    input.loc[0, "OverallQual"] = st.number_input("OverallQual: Overall material and finish quality")
    input.loc[0, "OverallCond"] = st.number_input("OverallCond: Overall condition rating")
    input.loc[0, "YearBuilt"] = st.number_input("YearBuilt: Original construction date")
    input.loc[0, "YearRemodAdd"] = st.number_input("YearRemodAdd: Remodel date")
    input.loc[0, "MasVnrArea"] = st.number_input("MasVnrArea: Masonry veneer area in square feet")
    input.loc[0, "BsmtFinSF1"] = st.number_input("BsmtFinSF1: Type 1 finished square feet")
    input.loc[0, "BsmtUnfSF"] = st.number_input("BsmtUnfSF: Unfinished square feet of basement area")
    input.loc[0, "TotalBsmtSF"] = st.number_input("TotalBsmtSF: Total square feet of basement area")
    input.loc[0, "1stFlrSF"] = st.number_input("1stFlrSF: First Floor square feet")
    input.loc[0, "2ndFlrSF"] = st.number_input("2ndFlrSF: Second floor square feet")
    input.loc[0, "GrLivArea"] = st.number_input("GrLivArea: Above grade (ground) living area square feet")
    input.loc[0, "BsmtFullBath"] = st.number_input("BsmtFullBath: Basement full bathrooms")
    input.loc[0, "BsmtHalfBath"] = st.number_input("BsmtHalfBath: Basement half bathrooms")
    input.loc[0, "FullBath"] = st.number_input("FullBath: Full bathrooms above grade")
    input.loc[0, "HalfBath"] = st.number_input("HalfBath: Half baths above grade")
    input.loc[0, "BedroomAbvGr"] = st.number_input("BedroomAbvGr: Number of bedrooms above basement level")
    input.loc[0, "KitchenAbvGr"] = st.number_input("KitchenAbvGr: Number of kitchens")
    input.loc[0, "TotRmsAbvGrd"] = st.number_input("TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)")
    input.loc[0, "Fireplaces"] = st.number_input("Fireplaces: Number of fireplaces")
    input.loc[0, "GarageYrBlt"] = st.number_input("GarageYrBlt: Year garage was built")
    input.loc[0, "GarageCars"] = st.number_input("GarageCars: Size of garage in car capacity")
    input.loc[0, "GarageArea"] = st.number_input("GarageArea: Size of garage in square feet")
    input.loc[0, "WoodDeckSF"] = st.number_input("WoodDeckSF: Wood deck area in square feet")
    input.loc[0, "OpenPorchSF"] = st.number_input("OpenPorchSF: Open porch area in square feet")


with c2:   
       st.subheader("Only input text values according to the metadata below")
       input.loc[0, "MSZoning"] = st.selectbox("MSZoning: The general zoning classification", (None, "A", "C", "FV", "I", "RH", "RL", "RP", "RM"))
       input.loc[0, "Street"] = st.selectbox("Street: Type of road access", (None, "Grvl", "Pave"))
       input.loc[0, "LotShape"] = st.selectbox("LotShape: General shape of property", (None, "Reg", "IR1", "IR2", "IR3"))
       input.loc[0, "LandContour"] = st.selectbox("LandContour: Flatness of the property", (None, "Lvl", "Bnk", "HLS", "Low"))
       input.loc[0, "Utilities"] = st.selectbox("Utilities: Type of utilities available", (None, "AllPub", "NoSewr", "NoSeWa", "ELO"))
       input.loc[0, "LotConfig"] = st.selectbox("LotConfig: Lot configuration", (None, "Inside", "Corner", "CulDSac", "FR2", "FR3"))
       input.loc[0, "LandSlope"] = st.selectbox("LandSlope: Slope of property", (None, "Gtl", "Mod", "Sev"))
       input.loc[0, "Neighborhood"] = st.selectbox("Neighborhood: Physical locations within Ames city limits", (None, "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"))
       input.loc[0, "Condition1"] = st.selectbox("Condition1: Proximity to main road or railroad", (None, "Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"))
       input.loc[0, "Condition2"] = st.selectbox("Condition2: Proximity to main road or railroad (if a second is present)", (None, "Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"))
       input.loc[0, "BldgType"] = st.selectbox("BldgType: Type of dwelling", (None, "1Fam", "2FmCon", "Duplex", "TwnhsE", "Twnhs"))
       input.loc[0, "HouseStyle"] = st.selectbox("HouseStyle: Style of dwelling", (None, "1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"))
       input.loc[0, "RoofStyle"] = st.selectbox("RoofStyle: Type of roof", (None, "Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"))
       input.loc[0, "RoofMatl"] = st.selectbox("RoofMatl: Roof material", (None, "ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"))
       input.loc[0, "Exterior1st"] = st.selectbox("Exterior1st: Exterior covering on house", (None, "AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"))
       input.loc[0, "Exterior2nd"] = st.selectbox("Exterior2nd: Exterior covering on house (if more than one material)", (None, "AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"))
       input.loc[0, "MasVnrType"] = st.selectbox("MasVnrType: Masonry veneer type", (None, "BrkCmn", "BrkFace", "CBlock", "None", "Stone"))
       input.loc[0, "ExterQual"] = st.selectbox("ExterQual: Exterior material quality", (None, "Ex", "Gd", "TA", "Fa", "Po"))
       input.loc[0, "ExterCond"] = st.selectbox("ExterCond: Present condition of the material on the exterior", (None, "Ex", "Gd", "TA", "Fa", "Po"))
       input.loc[0, "Foundation"] = st.selectbox("Foundation: Type of foundation", (None, "BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"))
       input.loc[0, "BsmtQual"] = st.selectbox("BsmtQual: Height of the basement", (None, "Ex", "Gd", "TA", "Fa", "Po", "NA"))
       input.loc[0, "BsmtCond"] = st.selectbox("BsmtCond: General condition of the basement", (None, "Ex", "Gd", "TA", "Fa", "Po", "NA"))
       input.loc[0, "BsmtExposure"] = st.selectbox("BsmtExposure: Walkout or garden level basement walls", (None, "Gd", "Av", "Mn", "No", "NA"))
       input.loc[0, "BsmtFinType1"] = st.selectbox("BsmtFinType1: Quality of basement finished area", (None, "GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"))
       input.loc[0, "BsmtFinType2"] = st.selectbox("BsmtFinType2: Quality of second finished area (if present)", (None, "GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"))
       input.loc[0, "Heating"] = st.selectbox("Heating: Type of heating", (None, "Floor", "GasA", "GasW", "Grav", "OthW", "Wall"))
       input.loc[0, "HeatingQC"] = st.selectbox("HeatingQC: Heating quality and condition", (None, "Ex", "Gd", "TA", "Fa", "Po"))
       input.loc[0, "CentralAir"] = st.selectbox("CentralAir: Central air conditioning", (None, "Y", "N"))
       input.loc[0, "Electrical"] = st.selectbox("Electrical: Electrical system", (None, "SBrkr", "FuseA", "FuseF", "FuseP", "Mix"))
       input.loc[0, "KitchenQual"] = st.selectbox("KitchenQual: Kitchen quality", (None, "Ex", "Gd", "TA", "Fa", "Po"))
       input.loc[0, "Functional"] = st.selectbox("Functional: Home functionality rating", (None, "Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"))
       input.loc[0, "FireplaceQu"] = st.selectbox("FireplaceQu: Fireplace quality", (None, "Ex", "Gd", "TA", "Fa", "Po", "NA"))
       input.loc[0, "GarageType"] = st.selectbox("GarageType: Garage location", (None, "2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "NA"))
       input.loc[0, "GarageFinish"] = st.selectbox("GarageFinish: Interior finish of the garage", (None, "Fin", "RFn", "Unf", "NA"))
       input.loc[0, "GarageQual"] = st.selectbox("GarageQual: Garage quality", (None, "Ex", "Gd", "TA", "Fa", "Po", "NA"))
       input.loc[0, "GarageCond"] = st.selectbox("GarageCond: Garage condition", (None, "Ex", "Gd", "TA", "Fa", "Po", "NA"))
       input.loc[0, "PavedDrive"] = st.selectbox("PavedDrive: Paved driveway", (None, "Y", "P", "N"))
       input.loc[0, "SaleType"] = st.selectbox("SaleType: Type of sale", (None, "WD", "CWD", "VWD", "New", "COD", "Con", "ConLw", "ConLI", "ConLD", "Oth"))
       input.loc[0, "SaleCondition"] = st.selectbox("SaleCondition: Condition of sale", (None, "Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"))


st.text("")
st.markdown(
    """
    <style>
    .gold-banner {
        width: 100%; /* Full width of the page */
        height: 5px; /* Height of the banner */
        background-color: #eccd65; /* Gold color */
        text-align: center; /* Center-align text inside the banner */
        line-height: 5px; /* Vertically center text */
        font-size: 24px; /* Text size */
        color: black; /* Text color */
        font-weight: bold; /* Bold text */
    }
    </style>
    <div class="gold-banner"></div>
    """,
    unsafe_allow_html=True
)
st.text("")
#Prediction Process------------------------------------------------------------------------------------------------------------------------------------------------

#loading pipeline
pipeline = joblib.load('pipelinexgb.pkl')
#prediction
pred = pipeline.predict(input)

#Prediction Result Terminal------------------------------------------------------------------------------------------------------------------------------------
st.header("Prediction Result")
st.subheader("Summary of your property attributes")
st.dataframe(input)
st.subheader("estimate is " + str(pred))


st.text("")
st.markdown(
    """
    <style>
    .gold-banner {
        width: 100%; /* Full width of the page */
        height: 20px; /* Height of the banner */
        background-color: #eccd65; /* Gold color */
        text-align: center; /* Center-align text inside the banner */
        line-height: 75px; /* Vertically center text */
        font-size: 24px; /* Text size */
        color: black; /* Text color */
        font-weight: bold; /* Bold text */
    }
    </style>
    <div class="gold-banner"></div>
    """,
    unsafe_allow_html=True
)
st.text("")

#Data Description
cc1,cc2 = st.columns([1,1])
dtext  = """SalePrice - the property's sale price in dollars\n
MSSubClass: The building class\n
MSZoning: The general zoning classification\n
LotFrontage: Linear feet of street connected to property\n
LotArea: Lot size in square feet\n
Street: Type of road access\n
Alley: Type of alley access\n
LotShape: General shape of property\n
LandContour: Flatness of the property\n
Utilities: Type of utilities available\n
LotConfig: Lot configuration\n
LandSlope: Slope of property\n
Neighborhood: Physical locations within Ames city limits\n
Condition1: Proximity to main road or railroad\n
Condition2: Proximity to main road or railroad (if a second is present)\n
BldgType: Type of dwelling\n
HouseStyle: Style of dwelling\n
OverallQual: Overall material and finish quality\n
OverallCond: Overall condition rating\n
YearBuilt: Original construction date\n
YearRemodAdd: Remodel date\n
RoofStyle: Type of roof\n
RoofMatl: Roof material\n
Exterior1st: Exterior covering on house\n
Exterior2nd: Exterior covering on house (if more than one material)\n
MasVnrType: Masonry veneer type\n
MasVnrArea: Masonry veneer area in square feet\n
ExterQual: Exterior material quality\n
ExterCond: Present condition of the material on the exterior\n
Foundation: Type of foundation\n
BsmtQual: Height of the basement\n
BsmtCond: General condition of the basement\n
BsmtExposure: Walkout or garden level basement walls\n
BsmtFinType1: Quality of basement finished area\n
BsmtFinSF1: Type 1 finished square feet\n
BsmtFinType2: Quality of second finished area (if present)\n
BsmtFinSF2: Type 2 finished square feet\n
BsmtUnfSF: Unfinished square feet of basement area\n
TotalBsmtSF: Total square feet of basement area\n
Heating: Type of heating\n
HeatingQC: Heating quality and condition\n
CentralAir: Central air conditioning\n
Electrical: Electrical system\n
1stFlrSF: First Floor square feet\n
2ndFlrSF: Second floor square feet\n
LowQualFinSF: Low quality finished square feet (all floors)\n
GrLivArea: Above grade (ground) living area square feet\n
BsmtFullBath: Basement full bathrooms\n
BsmtHalfBath: Basement half bathrooms\n
FullBath: Full bathrooms above grade\n
HalfBath: Half baths above grade\n
Bedroom: Number of bedrooms above basement level\n
Kitchen: Number of kitchens\n
KitchenQual: Kitchen quality\n
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)\n
Functional: Home functionality rating\n
Fireplaces: Number of fireplaces\n
FireplaceQu: Fireplace quality\n
GarageType: Garage location\n
GarageYrBlt: Year garage was built\n
GarageFinish: Interior finish of the garage\n
GarageCars: Size of garage in car capacity\n
GarageArea: Size of garage in square feet\n
GarageQual: Garage quality\n
GarageCond: Garage condition\n
PavedDrive: Paved driveway\n
WoodDeckSF: Wood deck area in square feet\n
OpenPorchSF: Open porch area in square feet\n
EnclosedPorch: Enclosed porch area in square feet\n
3SsnPorch: Three season porch area in square feet\n
ScreenPorch: Screen porch area in square feet\n
PoolArea: Pool area in square feet\n
PoolQC: Pool quality\n
Fence: Fence quality\n
MiscFeature: Miscellaneous feature not covered in other categories\n
MiscVal: $Value of miscellaneous feature\n
MoSold: Month Sold\n
YrSold: Year Sold\n
SaleType: Type of sale\n
SaleCondition: Condition of sale\n"""

dtextlong = """ MSSubClass: Identifies the type of dwelling involved in the sale.	

        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES

MSZoning: Identifies the general zoning classification of the sale.
		
       A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park 
       RM	Residential Medium Density
	
LotFrontage: Linear feet of street connected to property

LotArea: Lot size in square feet

Street: Type of road access to property

       Grvl	Gravel	
       Pave	Paved
       	
Alley: Type of alley access to property

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
		
LotShape: General shape of property

       Reg	Regular	
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular
       
LandContour: Flatness of the property

       Lvl	Near Flat/Level	
       Bnk	Banked - Quick and significant rise from street grade to building
       HLS	Hillside - Significant slope from side to side
       Low	Depression
		
Utilities: Type of utilities available
		
       AllPub	All public Utilities (E,G,W,& S)	
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only
       ELO	Electricity only	
	
LotConfig: Lot configuration

       Inside	Inside lot
       Corner	Corner lot
       CulDSac	Cul-de-sac
       FR2	Frontage on 2 sides of property
       FR3	Frontage on 3 sides of property
	
LandSlope: Slope of property
		
       Gtl	Gentle slope
       Mod	Moderate Slope	
       Sev	Severe Slope
	
Neighborhood: Physical locations within Ames city limits

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker
			
Condition1: Proximity to various conditions
	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
Condition2: Proximity to various conditions (if more than one is present)
		
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
BldgType: Type of dwelling
		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit
	
HouseStyle: Style of dwelling
	
       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished
       2Story	Two story
       2.5Fin	Two and one-half story: 2nd level finished
       2.5Unf	Two and one-half story: 2nd level unfinished
       SFoyer	Split Foyer
       SLvl	Split Level
	
OverallQual: Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
	
OverallCond: Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
		
YearBuilt: Original construction date

YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

RoofStyle: Type of roof

       Flat	Flat
       Gable	Gable
       Gambrel	Gabrel (Barn)
       Hip	Hip
       Mansard	Mansard
       Shed	Shed
		
RoofMatl: Roof material

       ClyTile	Clay or Tile
       CompShg	Standard (Composite) Shingle
       Membran	Membrane
       Metal	Metal
       Roll	Roll
       Tar&Grv	Gravel & Tar
       WdShake	Wood Shakes
       WdShngl	Wood Shingles
		
Exterior1st: Exterior covering on house

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
Exterior2nd: Exterior covering on house (if more than one material)

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
MasVnrType: Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone
	
MasVnrArea: Masonry veneer area in square feet

ExterQual: Evaluates the quality of the material on the exterior 
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
ExterCond: Evaluates the present condition of the material on the exterior
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
Foundation: Type of foundation
		
       BrkTil	Brick & Tile
       CBlock	Cinder Block
       PConc	Poured Contrete	
       Slab	Slab
       Stone	Stone
       Wood	Wood
		
BsmtQual: Evaluates the height of the basement

       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement
		
BsmtCond: Evaluates the general condition of the basement

       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement
	
BsmtExposure: Refers to walkout or garden level walls

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement
	
BsmtFinType1: Rating of basement finished area

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
		
BsmtFinSF1: Type 1 finished square feet

BsmtFinType2: Rating of basement finished area (if multiple types)

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement

BsmtFinSF2: Type 2 finished square feet

BsmtUnfSF: Unfinished square feet of basement area

TotalBsmtSF: Total square feet of basement area

Heating: Type of heating
		
       Floor	Floor Furnace
       GasA	Gas forced warm air furnace
       GasW	Gas hot water or steam heat
       Grav	Gravity furnace	
       OthW	Hot water or steam heat other than gas
       Wall	Wall furnace
		
HeatingQC: Heating quality and condition

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
CentralAir: Central air conditioning

       N	No
       Y	Yes
		
Electrical: Electrical system

       SBrkr	Standard Circuit Breakers & Romex
       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       Mix	Mixed
		
1stFlrSF: First Floor square feet
 
2ndFlrSF: Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea: Above grade (ground) living area square feet

BsmtFullBath: Basement full bathrooms

BsmtHalfBath: Basement half bathrooms

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

Bedroom: Bedrooms above grade (does NOT include basement bedrooms)

Kitchen: Kitchens above grade

KitchenQual: Kitchen quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       	
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

Functional: Home functionality (Assume typical unless deductions are warranted)

       Typ	Typical Functionality
       Min1	Minor Deductions 1
       Min2	Minor Deductions 2
       Mod	Moderate Deductions
       Maj1	Major Deductions 1
       Maj2	Major Deductions 2
       Sev	Severely Damaged
       Sal	Salvage only
		
Fireplaces: Number of fireplaces

FireplaceQu: Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
		
GarageType: Garage location
		
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage
		
GarageYrBlt: Year garage was built
		
GarageFinish: Interior finish of the garage

       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage
		
GarageCars: Size of garage in car capacity

GarageArea: Size of garage in square feet

GarageQual: Garage quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
GarageCond: Garage condition

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
PavedDrive: Paved driveway

       Y	Paved 
       P	Partial Pavement
       N	Dirt/Gravel
		
WoodDeckSF: Wood deck area in square feet

OpenPorchSF: Open porch area in square feet

EnclosedPorch: Enclosed porch area in square feet

3SsnPorch: Three season porch area in square feet

ScreenPorch: Screen porch area in square feet

PoolArea: Pool area in square feet

PoolQC: Pool quality
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool
		
Fence: Fence quality
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence
	
MiscFeature: Miscellaneous feature not covered in other categories
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None
		
MiscVal: $Value of miscellaneous feature

MoSold: Month Sold (MM)

YrSold: Year Sold (YYYY)

SaleType: Type of sale
		
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other
		
SaleCondition: Condition of sale

       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)
 """
with cc1: 
    st.header("Data description")
    st.text_area("", value= dtext, height = 500,  )
with cc2: 
    st.header("Metadata")
    st.text_area("", value= dtextlong, height= 500)



