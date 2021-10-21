#### Label Description
This dataset contains the temporal network of contacts between patients, patients and health-care workers (HCWs) and among HCWs in a hospital ward in Lyon, France, from Monday, December 6, 2010 at 1:00 pm to Friday, December 10, 2010 at 2:00 pm. The study included 46 HCWs and 29 patients.

The file contains a tab-separated list representing the active contacts during 20-second intervals of the data collection. Each line has the form “t i j Si Sj“, where i and j are the anonymous IDs of the persons in contact, Si and Sj are their statuses (NUR=paramedical staff, i.e. nurses and nurses’ aides; PAT=Patient; MED=Medical doctor; ADM=administrative staff), and the interval during which this contact was active is [ t – 20s, t ]. If multiple contacts are active in a given interval, you will see multiple lines starting with the same value of t. Time is measured in seconds.

'MED': 0, 'ADM': 1, 'NUR': 2, 'PAT': 3
NUR=paramedical staff, i.e. nurses and nurses’ aides; PAT=Patient; MED=Medical doctor; ADM=administrative staff
