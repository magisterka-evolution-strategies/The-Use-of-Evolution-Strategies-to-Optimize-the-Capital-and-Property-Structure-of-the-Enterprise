import sqlite3

filename = "../exporter_new.db"

conn = sqlite3.connect(filename)

cursor = conn.execute(
    "SELECT C.ID, COUNT(*) AS Total FROM Company AS C JOIN AssetsCategories AS AC ON AC.CompanyID = C.ID JOIN "
    "EquityLiabilitiesCategories AS ELC ON ELC.CompanyID = C.ID AND ELC.Date = AC.Date JOIN MarketValues MV ON "
    "MV.CompanyID = C.ID AND MV.[Period end] = ELC.Date GROUP BY C.ID HAVING Total > 1"
)

companies_with_needed_data = [row[0] for row in cursor]

cursor = conn.execute(
    "SELECT C.ID, MV.[Period end], MV.[Market value], AC.[Non-current assets], AC.[Current assets], AC.[Assets "
    "held for sale and discontinuing operations], AC.[Called up capital], AC.[Own shares], ELC.[Equity "
    "shareholders of the parent], ELC.[Non-controlling interests], ELC.[Non-current liabilities], ELC.[Current "
    "liabilities], ELC.[Liabilities related to assets held for sale and discontinued operations] FROM Company AS "
    "C JOIN AssetsCategories AS AC ON AC.CompanyID = C.ID JOIN EquityLiabilitiesCategories AS ELC ON "
    "ELC.CompanyID = C.ID AND ELC.Date = AC.Date JOIN MarketValues MV ON MV.CompanyID = C.ID AND MV.[Period end] "
    "= ELC.Date WHERE C.ID IN ({seq}) ORDER BY C.ID, MV.[Period end]".format(
        seq=','.join(['?'] * len(companies_with_needed_data))),
    companies_with_needed_data
)

data = [row for row in cursor]

conn.close()

print(len(data))


# cursor = conn.execute(
#     "SELECT C.ID, COUNT(*) AS Total FROM Company AS C JOIN AssetsCategories AS AC ON AC.CompanyID = C.ID JOIN "
#     "EquityLiabilitiesCategories AS ELC ON ELC.CompanyID = C.ID AND julianday(ELC.Date) BETWEEN julianday(AC.Date) - 30 AND julianday(AC.Date) + 30 JOIN MarketValues MV ON "
#     "MV.CompanyID = C.ID AND julianday(MV.[Period end]) BETWEEN julianday(ELC.Date) - 30 AND julianday(ELC.Date) + 30 GROUP BY C.ID HAVING Total > 1"
# )
# 
# companies_with_needed_data = [row[0] for row in cursor]
# 
# cursor = conn.execute(
#     "SELECT C.ID, MV.[Period end], MV.[Market value], AC.[Non-current assets], AC.[Current assets], AC.[Assets "
#     "held for sale and discontinuing operations], AC.[Called up capital], AC.[Own shares], ELC.[Equity "
#     "shareholders of the parent], ELC.[Non-controlling interests], ELC.[Non-current liabilities], ELC.[Current "
#     "liabilities], ELC.[Liabilities related to assets held for sale and discontinued operations] FROM Company AS "
#     "C JOIN AssetsCategories AS AC ON AC.CompanyID = C.ID JOIN EquityLiabilitiesCategories AS ELC ON "
#     "ELC.CompanyID = C.ID AND julianday(ELC.Date) BETWEEN julianday(AC.Date) - 30 AND julianday(AC.Date) + 30 JOIN MarketValues MV ON MV.CompanyID = C.ID AND "
#     "julianday(MV.[Period end]) BETWEEN julianday(ELC.Date) - 30 AND julianday(ELC.Date) + 30 WHERE C.ID IN ({seq}) ORDER BY C.ID, MV.[Period end]".format(
#         seq=','.join(['?'] * len(companies_with_needed_data))),
#     companies_with_needed_data
# )
# 
# data = [row for row in cursor]
# 
# conn.close()
# 
# print(len(data))


# cursor = conn.execute(
#     "SELECT * FROM AssetsCategories"
# )
#
# companies_with_needed_data = [row for row in cursor]
#
#
# min_array = min(companies_with_needed_data, key=lambda x: datetime.strptime(x[1], "%Y-%m-%d"))
# max_array = max(companies_with_needed_data, key=lambda x: datetime.strptime(x[1], "%Y-%m-%d"))
#
# print("Array with min date:", min_array)
# print("Array with max date:", max_array)
#
# conn.close()


# cursor = conn.execute(
#     "SELECT * FROM Assets"
# )
# companies_with_needed_data = [row for row in cursor]
#
#
# # min_array = min(companies_with_needed_data, key=lambda x: datetime.strptime(x[1], "%Y-%m-%d"))
# # max_array = max(companies_with_needed_data, key=lambda x: datetime.strptime(x[1], "%Y-%m-%d"))
#
# print("Array with min date:", companies_with_needed_data[0][2:len(companies_with_needed_data[0]) - 1])
#
# conn.close()