import pandas as pd
from datetime import datetime

data = {
    'Date': ["Time"],
    'Data': ["Formaldehyde"]
}
 
df = pd.DataFrame(data)
 
df.to_csv('HCOH.csv', mode='a', index=False, header=False)
