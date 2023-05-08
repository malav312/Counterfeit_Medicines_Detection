import sqlite3

# Connect to the database file
conn = sqlite3.connect('removedwords.db')
c = conn.cursor()

# Select all records from the 'text' table
try:
    c.execute("SELECT TEXT FROM text")
    rows = c.fetchall()
    count = 0
    # Loop through the records and update them
    for row in rows:
        # Check if the word 'tablet' or 'capsule' appears at the end of the string
        if row[0].endswith('tablet') or row[0].endswith('capsule'):
            # Remove the word from the end of the string
            new_value = row[0][:-len('tablet')] if row[0].endswith('tablet') else row[0][:-len('capsule')]
            # Update the record with the new value
            c.execute("UPDATE text SET TEXT = ? WHERE TEXT = ?", (new_value, row[0]))
            count += 1
            print(count)

    # Commit the changes to the database
    conn.commit()

except sqlite3.Error as e:
    print("An error occurred:", e.args[0])

# Close the database connection
conn.close()
