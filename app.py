from flask import Flask, render_template, request, make_response
from werkzeug.utils import secure_filename
import os
try:
    #This file contains the function for resume converting and preprocessing (Built by Moyyn Developer, needs to be included in the project folder)
    from open_convert_and_clean_pdf import *
except:
    os.system('pip install tika')
    os.system('pip install num2words')
    from open_convert_and_clean_pdf import *
try:
    #This is for calculating Cosine_Similarity, Sklearn library might need to be installed with "pip install sklearn
    from sklearn.metrics.pairwise import cosine_similarity
    #This is fo vectorizing the documents (converting into a TF-IDF vector)
    from sklearn.feature_extraction.text import TfidfVectorizer
except:
    os.system('pip install sklearn')
    #This is for calculating Cosine_Similarity, Sklearn library might need to be installed with "pip install sklearn
    from sklearn.metrics.pairwise import cosine_similarity
    #This is fo vectorizing the documents (converting into a TF-IDF vector)
    from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd #library for data manipulation and analysis
#The next two libraries for calculating more advanced similarity measurements (Built by Moyyn Developer, needs to be included in the project folder)
try:
    from isc_similarity import isc_similarity
    from sqrtcos_similarity import sqrtcos_similarity
except:
    os.system('pip install tika')
    os.system('pip install num2words')
    from isc_similarity import isc_similarity
    from sqrtcos_similarity import sqrtcos_similarity

app = Flask(__name__)

from markupsafe import Markup
excel_upload_form = Markup("""
      <h2>Second: Select candidates  file (.csv) </h2>
      <form action = "/upload_candidates" method = "POST" accept=".csv, .xlsx" enctype = "multipart/form-data" >
         <div><input type = "file" name = "ca_file" /></div>
         </br>
         <div><input class="moyynButton--main" type = "submit" value="upload candidates file"/></div>
      </form>
      </br>
""")
results_download_form = Markup("""
        <h2>Third: Download results as "results.csv" </h2>
         <form action = "/download_results" enctype = "multipart/form-data">
         <div><input class="moyynButton--main" type = "submit" value="download reults file"/><div>
         </form>
""")

job_des=''
profiles ={}
results = pd.DataFrame()

@app.route('/')
def main_page():
    return render_template('upload.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
            # try:
            #     global job_des
            #     f = request.files['jd_file']
            #     fsn = secure_filename(f.filename)
            #     f.save(fsn)
            #     job_des = read_JD(fsn)
            #     job_des = cleanText(job_des)
            #     job_des = preprocess(job_des)
            #     return render_template('upload.html', job_des=job_des, excel_upload_form= excel_upload_form)
            # except:
            global job_des
            job_des = request.form.get('jd_text')
            job_des = cleanText(job_des)
            job_des = preprocess(job_des)

            return render_template('upload.html', job_des=job_des, excel_upload_form= excel_upload_form)

# def read_JD(f):
#     ext = os.path.splitext(f)[1]
#     if (ext=='.pdf'):
#         job_des = open(f, 'rb')
#         job_des = pdftotext_converter(job_des)
#         return job_des
#     else:
#         return ext

@app.route('/upload_candidates', methods=['GET', 'POST'])
def upload_candidates():
  if request.method == 'POST':
    f = request.files['ca_file']
    fsn = secure_filename(f.filename)
    f.save(fsn)
    global profiles
    candidate_name_column = "Name"  # change to the index of the candidate_name column in the excel file
    #candidate_headline_column = "About"  # change to the index of the candidate_headline column
    candidates_df = pd.read_csv(fsn)
    headers = list(candidates_df)
    headers = list(set(headers)-set(["Name", "Email", "CV", "Email ID", "Gender", "First Name", "Last Name", "CV_German", "Driver's License", "Family Status", "Relocation Willingness_x", "Feedback" , "Matching Percentage", "Total Keywords", "Language Score", "Remote Working Willingness", "Suggestion Type", "Notice Period", "Location Flag", "Earliest Joining Date", "Creation Timestamp"]))
    print(headers)
    # Iterate over all rows in the dataframe (csv file)
    for index, candidate in candidates_df.iterrows():
        content =[]
        content.append([candidate[col] for col in headers])  # extract the content(headline) ) of each row(candidate) as a plain text
        content=''.join(str (e) for e in content)
        try:
            content = cleanText(content)  # clean text
            content = content.replace('nan', '')
        except:
            content = "This is an invalid content"
        content = preprocess(content)  # preprocess  text
        profiles[candidate[candidate_name_column]] = content  # add the cleaned and preprocessed text into the profiles list (dictionary)
    return compare()
  else:
      return False

def compare():
    df = pd.DataFrame()
    for key, headline in profiles.items():
        merged_text = [headline, job_des]  # stores the job_description/headline pairs
        tf_idf_vectorizer = TfidfVectorizer()
        count_martix = tf_idf_vectorizer.fit_transform(merged_text)  # vectorizing the pair usinf TF-IDF
        # the following lines will calculate the similarity using three different methods and round the results
        Cos_matchPercentage = round(cosine_similarity(count_martix)[0][1] * 100, 3)
        SqrtCos_matchPercentage = round(sqrtcos_similarity(count_martix)[0][1] * 100, 3)
        ISC_matchPercentage = round(isc_similarity(count_martix)[0][1] * 100, 3)

        # the following will add rows to the dataframe containing name of candidate with corresponding similarity scores
        #new_row = {'Cos_score': Cos_matchPercentage,'SC_score': SqrtCos_matchPercentage,'ISC_score': ISC_matchPercentage,'name': key }
        new_row = pd.DataFrame({ 'Name': [key],'ISC_score': [ISC_matchPercentage],'Cos_score': [Cos_matchPercentage],'SC_score': [SqrtCos_matchPercentage] })
        #new_row = new_row [["name", "ISC_score", "SC_score"]]
        df = df.append(new_row, ignore_index=True)

    df.sort_values(by=['ISC_score'], inplace=True,ascending=False)  # sorting profiles according to the ISC (Improved Sqrt Cosine) Similarity Score
    df.reset_index(drop=True, inplace=True)
    global results
    results = df
    #results = results["Name", "ISC_score", "SC_score"]
    return render_template('upload.html', tables=[results.head(10).to_html(classes='data')], titles=results.columns.values, excel_upload_form=excel_upload_form, results_download_form=results_download_form)

@app.route('/download_results')
def download_results():
    resp = make_response(results.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=results.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

if __name__ == '__main__':
    app.run(debug=True)