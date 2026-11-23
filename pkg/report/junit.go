package report

import (
	"encoding/xml"
	"fmt"
	"os"
	"time"

	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
	"github.com/alexandrmotologa/hnswprobe/pkg/topology"
)

// JUnitTestSuite represents the top-level JUnit XML element.
type JUnitTestSuite struct {
	XMLName    xml.Name        `xml:"testsuite"`
	Name       string          `xml:"name,attr"`
	Tests      int             `xml:"tests,attr"`
	Failures   int             `xml:"failures,attr"`
	Errors     int             `xml:"errors,attr"`
	Time       float64         `xml:"time,attr"`
	Timestamp  string          `xml:"timestamp,attr"`
	TestCases  []JUnitTestCase `xml:"testcase"`
}

// JUnitTestCase represents an individual test assertion in the report.
type JUnitTestCase struct {
	Name      string        `xml:"name,attr"`
	Classname string        `xml:"classname,attr"`
	Time      float64       `xml:"time,attr"`
	Failure   *JUnitFailure `xml:"failure,omitempty"`
}

// JUnitFailure describes a failed assertion.
type JUnitFailure struct {
	Message string `xml:"message,attr"`
	Type    string `xml:"type,attr"`
	Content string `xml:",chardata"`
}

// GenerateJUnitReport formats vector health metrics as a JUnit XML document.
func GenerateJUnitReport(
	evalReport groundtruth.EvaluationReport,
	topoReport topology.GraphReport,
	minRecall float64,
	maxTombstoneRatio float64,
	duration time.Duration,
) JUnitTestSuite {
	suite := JUnitTestSuite{
		Name:      "hnswprobe.vector_health",
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Time:      duration.Seconds(),
	}

	testCases := make([]JUnitTestCase, 0)
	failures := 0

	// Test 1: Recall threshold
	tcRecall := JUnitTestCase{
		Name:      "TestMeanRecallAtK",
		Classname: "hnswprobe.recall",
		Time:      duration.Seconds() * 0.4,
	}
	if evalReport.MeanRecall < minRecall {
		failures++
		tcRecall.Failure = &JUnitFailure{
			Message: fmt.Sprintf("Recall %.4f is below minimum requirement %.4f", evalReport.MeanRecall, minRecall),
			Type:    "RecallDecayFailure",
			Content: fmt.Sprintf("Measured mean recall@%d was %.4f (min: %.4f, p50: %.4f, p95: %.4f)",
				evalReport.K, evalReport.MeanRecall, evalReport.MinRecall, evalReport.P50Recall, evalReport.P95Recall),
		}
	}
	testCases = append(testCases, tcRecall)

	// Test 2: Tombstone bloat ratio
	tcTomb := JUnitTestCase{
		Name:      "TestTombstoneBloatRatio",
		Classname: "hnswprobe.topology",
		Time:      duration.Seconds() * 0.3,
	}
	if topoReport.TombstoneBloatRatio > maxTombstoneRatio {
		failures++
		tcTomb.Failure = &JUnitFailure{
			Message: fmt.Sprintf("Tombstone ratio %.4f exceeds maximum limit %.4f", topoReport.TombstoneBloatRatio, maxTombstoneRatio),
			Type:    "TombstoneBloatFailure",
			Content: fmt.Sprintf("Index has %d dead nodes out of %d total nodes (TBR: %.4f)",
				topoReport.TombstoneNodes, topoReport.TotalNodes, topoReport.TombstoneBloatRatio),
		}
	}
	testCases = append(testCases, tcTomb)

	// Test 3: Orphan nodes ratio
	tcOrphan := JUnitTestCase{
		Name:      "TestOrphanRatio",
		Classname: "hnswprobe.connectivity",
		Time:      duration.Seconds() * 0.3,
	}
	if topoReport.OrphanRatio > 0.05 {
		failures++
		tcOrphan.Failure = &JUnitFailure{
			Message: fmt.Sprintf("Orphan ratio %.4f exceeds 5%% threshold", topoReport.OrphanRatio),
			Type:    "OrphanDisconnectionFailure",
			Content: fmt.Sprintf("%d orphan nodes detected in layer 0 with in-degree 0", topoReport.OrphanNodesCount),
		}
	}
	testCases = append(testCases, tcOrphan)

	suite.Tests = len(testCases)
	suite.Failures = failures
	suite.TestCases = testCases

	return suite
}

// WriteJUnitReport exports the test suite to a file.
func WriteJUnitReport(path string, suite JUnitTestSuite) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	if _, err := file.WriteString(xml.Header); err != nil {
		return err
	}

	encoder := xml.NewEncoder(file)
	encoder.Indent("", "  ")
	return encoder.Encode(suite)
}
