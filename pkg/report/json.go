package report

import (
	"encoding/json"
	"os"
)

// WriteJSONReport saves an audit data structure to a JSON file.
func WriteJSONReport(path string, data any) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}
