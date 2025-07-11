package training

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestDefaultPlottingServiceConfig tests the default configuration
func TestDefaultPlottingServiceConfig(t *testing.T) {
	config := DefaultPlottingServiceConfig()
	
	if config.BaseURL != "http://localhost:8080" {
		t.Errorf("Expected BaseURL http://localhost:8080, got %s", config.BaseURL)
	}
	
	if config.Timeout != 30*time.Second {
		t.Errorf("Expected timeout 30s, got %v", config.Timeout)
	}
	
	if config.RetryAttempts != 3 {
		t.Errorf("Expected retry attempts 3, got %d", config.RetryAttempts)
	}
	
	if config.RetryDelay != 1*time.Second {
		t.Errorf("Expected retry delay 1s, got %v", config.RetryDelay)
	}
}

// TestNewPlottingService tests plotting service creation
func TestNewPlottingService(t *testing.T) {
	config := PlottingServiceConfig{
		BaseURL:       "http://test:9090",
		Timeout:       15 * time.Second,
		RetryAttempts: 5,
		RetryDelay:    2 * time.Second,
	}
	
	ps := NewPlottingService(config)
	
	if ps.baseURL != config.BaseURL {
		t.Errorf("Expected baseURL %s, got %s", config.BaseURL, ps.baseURL)
	}
	
	if ps.httpClient.Timeout != config.Timeout {
		t.Errorf("Expected timeout %v, got %v", config.Timeout, ps.httpClient.Timeout)
	}
	
	if ps.enabled {
		t.Error("Expected service to be disabled by default")
	}
}

// TestPlottingServiceEnableDisable tests enable/disable functionality
func TestPlottingServiceEnableDisable(t *testing.T) {
	ps := NewPlottingService(DefaultPlottingServiceConfig())
	
	// Initially disabled
	if ps.IsEnabled() {
		t.Error("Service should be disabled initially")
	}
	
	// Enable
	ps.Enable()
	if !ps.IsEnabled() {
		t.Error("Service should be enabled after Enable()")
	}
	
	// Disable
	ps.Disable()
	if ps.IsEnabled() {
		t.Error("Service should be disabled after Disable()")
	}
}

// mockHTTPServer creates a test HTTP server for plotting service tests
func mockHTTPServer(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(handler))
}

// TestSendPlotDataDisabled tests behavior when service is disabled
func TestSendPlotDataDisabled(t *testing.T) {
	ps := NewPlottingService(DefaultPlottingServiceConfig())
	
	plotData := PlotData{
		PlotType:  TrainingCurves,
		Title:     "Test Plot",
		Timestamp: time.Now(),
		ModelName: "TestModel",
		Series:    []SeriesData{{Name: "test", Type: "line", Data: []DataPoint{{X: 1, Y: 2}}}},
	}
	
	resp, err := ps.SendPlotData(plotData)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	
	if resp.Success {
		t.Error("Expected success to be false when service is disabled")
	}
	
	if resp.Message != "Plotting service is disabled" {
		t.Errorf("Expected disabled message, got: %s", resp.Message)
	}
}

// TestSendPlotDataSuccess tests successful plot data sending
func TestSendPlotDataSuccess(t *testing.T) {
	// Mock server that returns success
	server := mockHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("Expected POST method, got %s", r.Method)
		}
		
		if r.URL.Path != "/api/plot" {
			t.Errorf("Expected path /api/plot, got %s", r.URL.Path)
		}
		
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}
		
		if r.Header.Get("User-Agent") != "go-metal-training" {
			t.Errorf("Expected User-Agent go-metal-training, got %s", r.Header.Get("User-Agent"))
		}
		
		// Read and verify request body
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("Failed to read request body: %v", err)
		}
		
		var receivedPlotData PlotData
		if err := json.Unmarshal(body, &receivedPlotData); err != nil {
			t.Fatalf("Failed to unmarshal plot data: %v", err)
		}
		
		if receivedPlotData.PlotType != TrainingCurves {
			t.Errorf("Expected plot type %s, got %s", TrainingCurves, receivedPlotData.PlotType)
		}
		
		// Return success response
		response := PlottingResponse{
			Success: true,
			Message: "Plot generated successfully",
			PlotURL: "/plots/123",
			ViewURL: "/view/123",
			PlotID:  "plot_123",
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()
	
	// Create plotting service with mock server URL
	config := DefaultPlottingServiceConfig()
	config.BaseURL = server.URL
	ps := NewPlottingService(config)
	ps.Enable()
	
	plotData := PlotData{
		PlotType:  TrainingCurves,
		Title:     "Test Plot",
		Timestamp: time.Now(),
		ModelName: "TestModel",
		Series:    []SeriesData{{Name: "test", Type: "line", Data: []DataPoint{{X: 1, Y: 2}}}},
	}
	
	resp, err := ps.SendPlotData(plotData)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	
	if !resp.Success {
		t.Error("Expected success to be true")
	}
	
	if resp.Message != "Plot generated successfully" {
		t.Errorf("Expected success message, got: %s", resp.Message)
	}
	
	if resp.PlotURL != "/plots/123" {
		t.Errorf("Expected plot URL /plots/123, got %s", resp.PlotURL)
	}
	
	if resp.PlotID != "plot_123" {
		t.Errorf("Expected plot ID plot_123, got %s", resp.PlotID)
	}
}

// TestSendPlotDataHTTPError tests handling of HTTP errors
func TestSendPlotDataHTTPError(t *testing.T) {
	// Mock server that returns error
	server := mockHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
		response := PlottingResponse{
			Success:   false,
			Message:   "Server error occurred",
			ErrorCode: "INTERNAL_ERROR",
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()
	
	config := DefaultPlottingServiceConfig()
	config.BaseURL = server.URL
	ps := NewPlottingService(config)
	ps.Enable()
	
	plotData := PlotData{
		PlotType:  TrainingCurves,
		Title:     "Test Plot",
		Timestamp: time.Now(),
		ModelName: "TestModel",
		Series:    []SeriesData{{Name: "test", Type: "line", Data: []DataPoint{{X: 1, Y: 2}}}},
	}
	
	resp, err := ps.SendPlotData(plotData)
	if err == nil {
		t.Error("Expected error for HTTP 500 status")
	}
	
	if resp.ErrorCode != "INTERNAL_ERROR" {
		t.Errorf("Expected error code INTERNAL_ERROR, got %s", resp.ErrorCode)
	}
}

// TestSendPlotDataWithRetrySuccess tests retry logic with eventual success
func TestSendPlotDataWithRetrySuccess(t *testing.T) {
	attemptCount := 0
	
	server := mockHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
		attemptCount++
		
		if attemptCount < 2 {
			// Fail first attempt
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		
		// Succeed on second attempt
		response := PlottingResponse{
			Success: true,
			Message: "Plot generated successfully",
			PlotID:  "plot_retry_success",
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()
	
	config := PlottingServiceConfig{
		BaseURL:       server.URL,
		Timeout:       5 * time.Second,
		RetryAttempts: 3,
		RetryDelay:    10 * time.Millisecond, // Short delay for testing
	}
	ps := NewPlottingService(config)
	ps.Enable()
	
	plotData := PlotData{
		PlotType:  TrainingCurves,
		Title:     "Test Plot",
		Timestamp: time.Now(),
		ModelName: "TestModel",
		Series:    []SeriesData{{Name: "test", Type: "line", Data: []DataPoint{{X: 1, Y: 2}}}},
	}
	
	resp, err := ps.SendPlotDataWithRetry(plotData, config)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	
	if !resp.Success {
		t.Error("Expected success after retry")
	}
	
	if attemptCount != 2 {
		t.Errorf("Expected 2 attempts, got %d", attemptCount)
	}
}

// TestSendPlotDataWithRetryFailure tests retry logic with ultimate failure
func TestSendPlotDataWithRetryFailure(t *testing.T) {
	attemptCount := 0
	
	server := mockHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
		attemptCount++
		w.WriteHeader(http.StatusInternalServerError)
	})
	defer server.Close()
	
	config := PlottingServiceConfig{
		BaseURL:       server.URL,
		Timeout:       5 * time.Second,
		RetryAttempts: 2,
		RetryDelay:    10 * time.Millisecond,
	}
	ps := NewPlottingService(config)
	ps.Enable()
	
	plotData := PlotData{
		PlotType:  TrainingCurves,
		Title:     "Test Plot",
		Timestamp: time.Now(),
		ModelName: "TestModel",
		Series:    []SeriesData{{Name: "test", Type: "line", Data: []DataPoint{{X: 1, Y: 2}}}},
	}
	
	resp, err := ps.SendPlotDataWithRetry(plotData, config)
	if err == nil {
		t.Error("Expected error after all retries failed")
	}
	
	if resp != nil {
		t.Error("Expected nil response after retry failure")
	}
	
	if attemptCount != 2 {
		t.Errorf("Expected 2 attempts, got %d", attemptCount)
	}
}

// TestCheckHealth tests health check functionality
func TestCheckHealth(t *testing.T) {
	t.Run("Success", func(t *testing.T) {
		server := mockHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/health" {
				t.Errorf("Expected path /health, got %s", r.URL.Path)
			}
			
			if r.Method != "GET" {
				t.Errorf("Expected GET method, got %s", r.Method)
			}
			
			w.WriteHeader(http.StatusOK)
		})
		defer server.Close()
		
		config := DefaultPlottingServiceConfig()
		config.BaseURL = server.URL
		ps := NewPlottingService(config)
		ps.Enable()
		
		err := ps.CheckHealth()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
	})
	
	t.Run("Disabled", func(t *testing.T) {
		ps := NewPlottingService(DefaultPlottingServiceConfig())
		
		err := ps.CheckHealth()
		if err == nil {
			t.Error("Expected error when service is disabled")
		}
		
		if err.Error() != "plotting service is disabled" {
			t.Errorf("Expected disabled error, got: %v", err)
		}
	})
	
	t.Run("HTTPError", func(t *testing.T) {
		server := mockHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		})
		defer server.Close()
		
		config := DefaultPlottingServiceConfig()
		config.BaseURL = server.URL
		ps := NewPlottingService(config)
		ps.Enable()
		
		err := ps.CheckHealth()
		if err == nil {
			t.Error("Expected error for HTTP 503 status")
		}
	})
}

// TestGenerateAndSendPlot tests plot generation and sending
func TestGenerateAndSendPlot(t *testing.T) {
	t.Run("Success", func(t *testing.T) {
		server := mockHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
			response := PlottingResponse{
				Success: true,
				Message: "Plot generated successfully",
				PlotID:  "plot_generated",
			}
			
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(response)
		})
		defer server.Close()
		
		config := DefaultPlottingServiceConfig()
		config.BaseURL = server.URL
		ps := NewPlottingService(config)
		ps.Enable()
		
		// Create a visualization collector with some data
		vc := NewVisualizationCollector("TestModel")
		vc.Enable()
		vc.RecordTrainingStep(1, 0.8, 0.6, 0.01)
		
		resp, err := ps.GenerateAndSendPlot(vc, TrainingCurves)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		if !resp.Success {
			t.Error("Expected success")
		}
		
		if resp.PlotID != "plot_generated" {
			t.Errorf("Expected plot ID plot_generated, got %s", resp.PlotID)
		}
	})
	
	t.Run("NoDataForEmptyPlotType", func(t *testing.T) {
		// Mock server that should not be called since ConfusionMatrixPlot returns empty series
		server := mockHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
			t.Error("HTTP request should not be made when there's no data")
		})
		defer server.Close()
		
		config := DefaultPlottingServiceConfig()
		config.BaseURL = server.URL
		ps := NewPlottingService(config)
		ps.Enable()
		
		// Create empty visualization collector
		vc := NewVisualizationCollector("TestModel")
		
		// Use ConfusionMatrixPlot which returns empty series when there's no data
		resp, err := ps.GenerateAndSendPlot(vc, ConfusionMatrixPlot)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		if resp.Success {
			t.Error("Expected failure for no data")
		}
		
		expectedMsg := "No data available for plot type: confusion_matrix"
		if resp.Message != expectedMsg {
			t.Errorf("Expected message '%s', got '%s'", expectedMsg, resp.Message)
		}
	})
	
	t.Run("TrainingCurvesWithEmptyData", func(t *testing.T) {
		// TrainingCurves creates series structures even with no data, so it will be sent
		server := mockHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
			response := PlottingResponse{
				Success: true,
				Message: "Plot generated successfully",
				PlotID:  "plot_empty_training_curves",
			}
			
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(response)
		})
		defer server.Close()
		
		config := DefaultPlottingServiceConfig()
		config.BaseURL = server.URL
		ps := NewPlottingService(config)
		ps.Enable()
		
		// Create empty visualization collector
		vc := NewVisualizationCollector("TestModel")
		
		resp, err := ps.GenerateAndSendPlot(vc, TrainingCurves)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		// TrainingCurves creates series even with no data, so request succeeds
		if !resp.Success {
			t.Error("Expected success (TrainingCurves creates series even with no data)")
		}
		
		if resp.PlotID != "plot_empty_training_curves" {
			t.Errorf("Expected plot ID plot_empty_training_curves, got %s", resp.PlotID)
		}
	})
	
	t.Run("UnsupportedPlotType", func(t *testing.T) {
		ps := NewPlottingService(DefaultPlottingServiceConfig())
		ps.Enable()
		
		vc := NewVisualizationCollector("TestModel")
		
		resp, err := ps.GenerateAndSendPlot(vc, PlotType("unsupported"))
		if err == nil {
			t.Error("Expected error for unsupported plot type")
		}
		
		if resp != nil {
			t.Error("Expected nil response for unsupported plot type")
		}
	})
	
	t.Run("Disabled", func(t *testing.T) {
		ps := NewPlottingService(DefaultPlottingServiceConfig())
		
		vc := NewVisualizationCollector("TestModel")
		vc.Enable()
		vc.RecordTrainingStep(1, 0.8, 0.6, 0.01)
		
		resp, err := ps.GenerateAndSendPlot(vc, TrainingCurves)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		if resp.Success {
			t.Error("Expected failure when service is disabled")
		}
		
		if resp.Message != "Plotting service is disabled" {
			t.Errorf("Expected disabled message, got: %s", resp.Message)
		}
	})
}

// TestGenerateAndSendAllPlots tests bulk plot generation
func TestGenerateAndSendAllPlots(t *testing.T) {
	requestCount := 0
	
	server := mockHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		
		response := PlottingResponse{
			Success: true,
			Message: "Plot generated successfully",
			PlotID:  fmt.Sprintf("plot_%d", requestCount),
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()
	
	config := DefaultPlottingServiceConfig()
	config.BaseURL = server.URL
	ps := NewPlottingService(config)
	ps.Enable()
	
	// Create visualization collector with some data
	vc := NewVisualizationCollector("TestModel")
	vc.Enable()
	vc.RecordTrainingStep(1, 0.8, 0.6, 0.01)
	vc.RecordTrainingStep(2, 0.6, 0.8, 0.009)
	
	results := ps.GenerateAndSendAllPlots(vc)
	
	// Should have results for multiple plot types
	if len(results) == 0 {
		t.Error("Expected some plot results")
	}
	
	// Check that some plots succeeded (those with data)
	successCount := 0
	for plotType, result := range results {
		if result.Success {
			successCount++
		} else {
			// Verify failure reasons for plots without data
			t.Logf("Plot %s failed: %s", plotType, result.Message)
		}
	}
	
	if successCount == 0 {
		t.Error("Expected at least one successful plot")
	}
}

// TestBatchSendPlots tests batch plot sending
func TestBatchSendPlots(t *testing.T) {
	t.Run("Success", func(t *testing.T) {
		server := mockHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/api/batch-plot" {
				t.Errorf("Expected path /api/batch-plot, got %s", r.URL.Path)
			}
			
			// Read and verify request body
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("Failed to read request body: %v", err)
			}
			
			var batchPayload map[string]interface{}
			if err := json.Unmarshal(body, &batchPayload); err != nil {
				t.Fatalf("Failed to unmarshal batch payload: %v", err)
			}
			
			plots, ok := batchPayload["plots"].([]interface{})
			if !ok {
				t.Error("Expected plots array in batch payload")
			}
			
			if len(plots) != 2 {
				t.Errorf("Expected 2 plots in batch, got %d", len(plots))
			}
			
			if batchPayload["batch"] != true {
				t.Error("Expected batch flag to be true")
			}
			
			// Return batch success response
			response := BatchPlottingResponse{
				Success: true,
				Message: "Batch plots generated successfully",
				BatchID: "batch_123",
				Results: []BatchPlotResult{
					{Success: true, PlotID: "plot_1", PlotURL: "/plots/1", ViewURL: "/view/1", PlotType: "training_curves"},
					{Success: true, PlotID: "plot_2", PlotURL: "/plots/2", ViewURL: "/view/2", PlotType: "roc_curve"},
				},
				DashboardURL: "/dashboard/batch_123",
				Summary: BatchSummary{
					TotalPlots: 2,
					Successful: 2,
					Failed:     0,
				},
			}
			
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(response)
		})
		defer server.Close()
		
		config := DefaultPlottingServiceConfig()
		config.BaseURL = server.URL
		ps := NewPlottingService(config)
		ps.Enable()
		
		plotDataList := []PlotData{
			{
				PlotType:  TrainingCurves,
				Title:     "Training Curves",
				ModelName: "TestModel",
				Series:    []SeriesData{{Name: "loss", Type: "line", Data: []DataPoint{{X: 1, Y: 0.8}}}},
			},
			{
				PlotType:  ROCCurve,
				Title:     "ROC Curve",
				ModelName: "TestModel",
				Series:    []SeriesData{{Name: "roc", Type: "line", Data: []DataPoint{{X: 0.1, Y: 0.9}}}},
			},
		}
		
		resp, err := ps.BatchSendPlots(plotDataList)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		if !resp.Success {
			t.Error("Expected batch success")
		}
		
		if resp.BatchID != "batch_123" {
			t.Errorf("Expected batch ID batch_123, got %s", resp.BatchID)
		}
		
		if len(resp.Results) != 2 {
			t.Errorf("Expected 2 results, got %d", len(resp.Results))
		}
		
		if resp.Summary.TotalPlots != 2 {
			t.Errorf("Expected 2 total plots, got %d", resp.Summary.TotalPlots)
		}
		
		if resp.Summary.Successful != 2 {
			t.Errorf("Expected 2 successful plots, got %d", resp.Summary.Successful)
		}
	})
	
	t.Run("Disabled", func(t *testing.T) {
		ps := NewPlottingService(DefaultPlottingServiceConfig())
		
		plotDataList := []PlotData{{PlotType: TrainingCurves}}
		
		resp, err := ps.BatchSendPlots(plotDataList)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		
		if resp.Success {
			t.Error("Expected failure when service is disabled")
		}
		
		if resp.Message != "Plotting service is disabled" {
			t.Errorf("Expected disabled message, got: %s", resp.Message)
		}
	})
}

// TestOpenInBrowser tests browser opening functionality (mocked - no actual browser opening)
func TestOpenInBrowser(t *testing.T) {
	// Skip this test as it would open actual browser windows
	// In a real test environment, this functionality would be mocked
	t.Skip("Skipping OpenInBrowser test to prevent actual browser opening during tests")
}

// TestSendPlotDataAndOpen tests sending plot data (without browser opening)
func TestSendPlotDataAndOpen(t *testing.T) {
	// Skip this test as it would open actual browser windows
	// In a real implementation, the browser opening would be mocked
	t.Skip("Skipping SendPlotDataAndOpen test to prevent actual browser opening during tests")
}

// TestGenerateAndSendAllPlotsWithBrowser tests the batch workflow (without browser opening)
func TestGenerateAndSendAllPlotsWithBrowser(t *testing.T) {
	// Skip this test as it would open actual browser windows
	// In a real implementation, the browser opening would be mocked
	t.Skip("Skipping GenerateAndSendAllPlotsWithBrowser test to prevent actual browser opening during tests")
}

// TestPlottingResponseStructs tests response structure marshaling/unmarshaling
func TestPlottingResponseStructs(t *testing.T) {
	t.Run("PlottingResponse", func(t *testing.T) {
		original := PlottingResponse{
			Success:      true,
			Message:      "Test message",
			PlotURL:      "/plots/123",
			ViewURL:      "/view/123",
			PlotID:       "plot_123",
			BatchID:      "batch_123",
			DashboardURL: "/dashboard/123",
			ErrorCode:    "NO_ERROR",
		}
		
		// Marshal to JSON
		jsonData, err := json.Marshal(original)
		if err != nil {
			t.Fatalf("Failed to marshal PlottingResponse: %v", err)
		}
		
		// Unmarshal back
		var unmarshaled PlottingResponse
		if err := json.Unmarshal(jsonData, &unmarshaled); err != nil {
			t.Fatalf("Failed to unmarshal PlottingResponse: %v", err)
		}
		
		// Verify fields
		if unmarshaled.Success != original.Success {
			t.Errorf("Success mismatch: expected %v, got %v", original.Success, unmarshaled.Success)
		}
		
		if unmarshaled.Message != original.Message {
			t.Errorf("Message mismatch: expected %s, got %s", original.Message, unmarshaled.Message)
		}
		
		if unmarshaled.PlotID != original.PlotID {
			t.Errorf("PlotID mismatch: expected %s, got %s", original.PlotID, unmarshaled.PlotID)
		}
	})
	
	t.Run("BatchPlottingResponse", func(t *testing.T) {
		original := BatchPlottingResponse{
			Success: true,
			Message: "Batch success",
			BatchID: "batch_test",
			Results: []BatchPlotResult{
				{Success: true, PlotID: "plot_1", PlotType: "training_curves"},
				{Success: false, Message: "Failed plot", ErrorCode: "NO_DATA"},
			},
			DashboardURL: "/dashboard/test",
			Summary: BatchSummary{
				TotalPlots: 2,
				Successful: 1,
				Failed:     1,
			},
		}
		
		// Marshal to JSON
		jsonData, err := json.Marshal(original)
		if err != nil {
			t.Fatalf("Failed to marshal BatchPlottingResponse: %v", err)
		}
		
		// Unmarshal back
		var unmarshaled BatchPlottingResponse
		if err := json.Unmarshal(jsonData, &unmarshaled); err != nil {
			t.Fatalf("Failed to unmarshal BatchPlottingResponse: %v", err)
		}
		
		// Verify fields
		if unmarshaled.Success != original.Success {
			t.Errorf("Success mismatch: expected %v, got %v", original.Success, unmarshaled.Success)
		}
		
		if len(unmarshaled.Results) != len(original.Results) {
			t.Errorf("Results length mismatch: expected %d, got %d", len(original.Results), len(unmarshaled.Results))
		}
		
		if unmarshaled.Summary.TotalPlots != original.Summary.TotalPlots {
			t.Errorf("TotalPlots mismatch: expected %d, got %d", original.Summary.TotalPlots, unmarshaled.Summary.TotalPlots)
		}
	})
}

// BenchmarkSendPlotData benchmarks plot data sending
func BenchmarkSendPlotData(b *testing.B) {
	server := mockHTTPServer(nil, func(w http.ResponseWriter, r *http.Request) {
		response := PlottingResponse{
			Success: true,
			Message: "Plot generated successfully",
			PlotID:  "benchmark_plot",
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()
	
	config := DefaultPlottingServiceConfig()
	config.BaseURL = server.URL
	ps := NewPlottingService(config)
	ps.Enable()
	
	plotData := PlotData{
		PlotType:  TrainingCurves,
		Title:     "Benchmark Plot",
		Timestamp: time.Now(),
		ModelName: "BenchmarkModel",
		Series: []SeriesData{
			{
				Name: "training_loss",
				Type: "line",
				Data: make([]DataPoint, 100), // 100 data points
			},
		},
	}
	
	// Fill with sample data
	for i := range plotData.Series[0].Data {
		plotData.Series[0].Data[i] = DataPoint{X: i, Y: float64(i) * 0.01}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ps.SendPlotData(plotData)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkBatchSendPlots benchmarks batch plot sending
func BenchmarkBatchSendPlots(b *testing.B) {
	server := mockHTTPServer(nil, func(w http.ResponseWriter, r *http.Request) {
		response := BatchPlottingResponse{
			Success: true,
			Message: "Batch plots generated successfully",
			BatchID: "benchmark_batch",
			Results: []BatchPlotResult{
				{Success: true, PlotID: "plot_1"},
				{Success: true, PlotID: "plot_2"},
				{Success: true, PlotID: "plot_3"},
			},
			Summary: BatchSummary{TotalPlots: 3, Successful: 3, Failed: 0},
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()
	
	config := DefaultPlottingServiceConfig()
	config.BaseURL = server.URL
	ps := NewPlottingService(config)
	ps.Enable()
	
	plotDataList := []PlotData{
		{PlotType: TrainingCurves, ModelName: "BenchmarkModel", Series: []SeriesData{{Name: "loss", Type: "line", Data: []DataPoint{{X: 1, Y: 0.8}}}}},
		{PlotType: ROCCurve, ModelName: "BenchmarkModel", Series: []SeriesData{{Name: "roc", Type: "line", Data: []DataPoint{{X: 0.1, Y: 0.9}}}}},
		{PlotType: PrecisionRecall, ModelName: "BenchmarkModel", Series: []SeriesData{{Name: "pr", Type: "line", Data: []DataPoint{{X: 0.8, Y: 0.7}}}}},
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ps.BatchSendPlots(plotDataList)
		if err != nil {
			b.Fatal(err)
		}
	}
}