package training

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"runtime"
	"time"
)

// PlottingService handles communication with the sidecar plotting application
type PlottingService struct {
	baseURL    string
	httpClient *http.Client
	enabled    bool
}

// PlottingServiceConfig contains configuration for the plotting service
type PlottingServiceConfig struct {
	BaseURL        string        `json:"base_url"`
	Timeout        time.Duration `json:"timeout"`
	RetryAttempts  int          `json:"retry_attempts"`
	RetryDelay     time.Duration `json:"retry_delay"`
}

// PlottingResponse represents the response from the plotting service
type PlottingResponse struct {
	Success      bool   `json:"success"`
	Message      string `json:"message"`
	PlotURL      string `json:"plot_url,omitempty"`
	ViewURL      string `json:"view_url,omitempty"`
	PlotID       string `json:"plot_id,omitempty"`
	BatchID      string `json:"batch_id,omitempty"`
	DashboardURL string `json:"dashboard_url,omitempty"`
	ErrorCode    string `json:"error_code,omitempty"`
}

// BatchPlottingResponse represents the response from the batch plotting endpoint
type BatchPlottingResponse struct {
	Success      bool                       `json:"success"`
	Message      string                     `json:"message"`
	BatchID      string                     `json:"batch_id,omitempty"`
	Results      []BatchPlotResult          `json:"results,omitempty"`
	DashboardURL string                     `json:"dashboard_url,omitempty"`
	Summary      BatchSummary               `json:"summary,omitempty"`
}

// BatchPlotResult represents a single plot result within a batch response
type BatchPlotResult struct {
	Success   bool   `json:"success"`
	PlotID    string `json:"plot_id,omitempty"`
	PlotURL   string `json:"plot_url,omitempty"`
	ViewURL   string `json:"view_url,omitempty"`
	PlotType  string `json:"plot_type,omitempty"`
	Message   string `json:"message,omitempty"`
	ErrorCode string `json:"error_code,omitempty"`
}

// BatchSummary represents the summary of a batch operation
type BatchSummary struct {
	TotalPlots int `json:"total_plots"`
	Successful int `json:"successful"`
	Failed     int `json:"failed"`
}

// DefaultPlottingServiceConfig returns default configuration for the plotting service
func DefaultPlottingServiceConfig() PlottingServiceConfig {
	return PlottingServiceConfig{
		BaseURL:       "http://localhost:8080",
		Timeout:       30 * time.Second,
		RetryAttempts: 3,
		RetryDelay:    1 * time.Second,
	}
}

// NewPlottingService creates a new plotting service client
func NewPlottingService(config PlottingServiceConfig) *PlottingService {
	return &PlottingService{
		baseURL: config.BaseURL,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
		enabled: false,
	}
}

// Enable enables the plotting service
func (ps *PlottingService) Enable() {
	ps.enabled = true
}

// Disable disables the plotting service
func (ps *PlottingService) Disable() {
	ps.enabled = false
}

// IsEnabled returns whether the plotting service is enabled
func (ps *PlottingService) IsEnabled() bool {
	return ps.enabled
}

// SendPlotData sends plot data to the sidecar plotting service
func (ps *PlottingService) SendPlotData(plotData PlotData) (*PlottingResponse, error) {
	if !ps.enabled {
		return &PlottingResponse{
			Success: false,
			Message: "Plotting service is disabled",
		}, nil
	}
	
	// Convert plot data to JSON
	jsonData, err := json.Marshal(plotData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal plot data: %w", err)
	}
	
	// Create HTTP request
	url := fmt.Sprintf("%s/api/plot", ps.baseURL)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}
	
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "go-metal-training")
	
	// Send request
	resp, err := ps.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send HTTP request: %w", err)
	}
	defer resp.Body.Close()
	
	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	
	// Parse response
	var plotResponse PlottingResponse
	if err := json.Unmarshal(respBody, &plotResponse); err != nil {
		return nil, fmt.Errorf("failed to parse response JSON: %w", err)
	}
	
	// Check HTTP status
	if resp.StatusCode != http.StatusOK {
		return &plotResponse, fmt.Errorf("HTTP request failed with status %d: %s", resp.StatusCode, plotResponse.Message)
	}
	
	return &plotResponse, nil
}

// SendPlotDataWithRetry sends plot data with retry logic
func (ps *PlottingService) SendPlotDataWithRetry(plotData PlotData, config PlottingServiceConfig) (*PlottingResponse, error) {
	if !ps.enabled {
		return &PlottingResponse{
			Success: false,
			Message: "Plotting service is disabled",
		}, nil
	}
	
	var lastErr error
	
	for attempt := 0; attempt < config.RetryAttempts; attempt++ {
		resp, err := ps.SendPlotData(plotData)
		if err == nil {
			return resp, nil
		}
		
		lastErr = err
		
		// Wait before retry (except for the last attempt)
		if attempt < config.RetryAttempts-1 {
			time.Sleep(config.RetryDelay)
		}
	}
	
	return nil, fmt.Errorf("failed to send plot data after %d attempts: %w", config.RetryAttempts, lastErr)
}

// CheckHealth checks if the plotting service is available
func (ps *PlottingService) CheckHealth() error {
	if !ps.enabled {
		return fmt.Errorf("plotting service is disabled")
	}
	
	url := fmt.Sprintf("%s/health", ps.baseURL)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}
	
	resp, err := ps.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send health check request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status %d", resp.StatusCode)
	}
	
	return nil
}

// GenerateAndSendPlot generates a plot and sends it to the sidecar service
func (ps *PlottingService) GenerateAndSendPlot(collector *VisualizationCollector, plotType PlotType) (*PlottingResponse, error) {
	if !ps.enabled {
		return &PlottingResponse{
			Success: false,
			Message: "Plotting service is disabled",
		}, nil
	}
	
	var plotData PlotData
	
	switch plotType {
	case TrainingCurves:
		plotData = collector.GenerateTrainingCurvesPlot()
	case LearningRateSchedule:
		plotData = collector.GenerateLearningRateSchedulePlot()
	case ROCCurve:
		plotData = collector.GenerateROCCurvePlot()
	case PrecisionRecall:
		plotData = collector.GeneratePrecisionRecallPlot()
	case ConfusionMatrixPlot:
		plotData = collector.GenerateConfusionMatrixPlot()
	case RegressionScatter:
		plotData = collector.GenerateRegressionScatterPlot()
	case ResidualPlot:
		plotData = collector.GenerateResidualPlot()
	case QQPlot:
		plotData = collector.GenerateQQPlot()
	case FeatureImportancePlot:
		plotData = collector.GenerateFeatureImportancePlot()
	case LearningCurvePlot:
		plotData = collector.GenerateLearningCurvePlot()
	case ValidationCurvePlot:
		plotData = collector.GenerateValidationCurvePlot()
	case PredictionIntervalPlot:
		plotData = collector.GeneratePredictionIntervalPlot()
	case FeatureCorrelationPlot:
		plotData = collector.GenerateFeatureCorrelationPlot()
	case PartialDependencePlot:
		plotData = collector.GeneratePartialDependencePlot()
	default:
		return nil, fmt.Errorf("unsupported plot type: %s", plotType)
	}
	
	// Check if plot data is valid
	if len(plotData.Series) == 0 {
		return &PlottingResponse{
			Success: false,
			Message: fmt.Sprintf("No data available for plot type: %s", plotType),
		}, nil
	}
	
	return ps.SendPlotData(plotData)
}

// GenerateAndSendAllPlots generates all available plots and sends them to the sidecar service
func (ps *PlottingService) GenerateAndSendAllPlots(collector *VisualizationCollector) map[PlotType]*PlottingResponse {
	results := make(map[PlotType]*PlottingResponse)
	
	if !ps.enabled {
		return results
	}
	
	// Define plot types to generate
	plotTypes := []PlotType{
		TrainingCurves,
		LearningRateSchedule,
		ROCCurve,
		PrecisionRecall,
		ConfusionMatrixPlot,
		RegressionScatter,
		ResidualPlot,
		QQPlot,
		FeatureImportancePlot,
		LearningCurvePlot,
		ValidationCurvePlot,
	}
	
	// Generate and send each plot type
	for _, plotType := range plotTypes {
		resp, err := ps.GenerateAndSendPlot(collector, plotType)
		if err != nil {
			results[plotType] = &PlottingResponse{
				Success: false,
				Message: err.Error(),
			}
		} else {
			results[plotType] = resp
		}
	}
	
	return results
}

// BatchSendPlots sends multiple plots in a single request
func (ps *PlottingService) BatchSendPlots(plotDataList []PlotData) (*BatchPlottingResponse, error) {
	if !ps.enabled {
		return &BatchPlottingResponse{
			Success: false,
			Message: "Plotting service is disabled",
		}, nil
	}
	
	// Create batch request payload
	batchPayload := map[string]interface{}{
		"plots": plotDataList,
		"batch": true,
	}
	
	// Convert to JSON
	jsonData, err := json.Marshal(batchPayload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal batch plot data: %w", err)
	}
	
	// Create HTTP request
	url := fmt.Sprintf("%s/api/batch-plot", ps.baseURL)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create batch HTTP request: %w", err)
	}
	
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "go-metal-training")
	
	// Send request
	resp, err := ps.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send batch HTTP request: %w", err)
	}
	defer resp.Body.Close()
	
	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read batch response body: %w", err)
	}
	
	// Debug: Print raw response
	// fmt.Printf("🔍 Debug: Raw batch response: %s\n", string(respBody))
	
	// Parse response as batch response
	var batchResponse BatchPlottingResponse
	if err := json.Unmarshal(respBody, &batchResponse); err != nil {
		return nil, fmt.Errorf("failed to parse batch response JSON: %w", err)
	}
	
	// Check HTTP status
	if resp.StatusCode != http.StatusOK {
		return &batchResponse, fmt.Errorf("batch HTTP request failed with status %d: %s", resp.StatusCode, batchResponse.Message)
	}
	
	return &batchResponse, nil
}

// OpenInBrowser opens the given URL in the default web browser
// It automatically detects the operating system and uses the appropriate command
func (ps *PlottingService) OpenInBrowser(url string) error {
	var cmd string
	var args []string
	
	switch runtime.GOOS {
	case "darwin": // macOS
		cmd = "open"
		args = []string{url}
	case "windows":
		cmd = "cmd"
		args = []string{"/c", "start", url}
	case "linux":
		// Try xdg-open first (most common), fallback to other options
		cmd = "xdg-open"
		args = []string{url}
	default:
		return fmt.Errorf("unsupported operating system: %s", runtime.GOOS)
	}
	
	err := exec.Command(cmd, args...).Start()
	if err != nil && runtime.GOOS == "linux" {
		// If xdg-open fails on Linux, try alternatives
		alternatives := []string{"gnome-open", "kde-open", "firefox", "google-chrome", "chromium"}
		for _, alt := range alternatives {
			if err = exec.Command(alt, url).Start(); err == nil {
				return nil
			}
		}
	}
	
	return err
}

// SendPlotDataAndOpen sends plot data and automatically opens the result in browser
func (ps *PlottingService) SendPlotDataAndOpen(plotData PlotData) (*PlottingResponse, error) {
	resp, err := ps.SendPlotData(plotData)
	if err != nil {
		return resp, err
	}
	
	if resp.Success {
		// Prefer ViewURL over PlotURL for better formatted display
		urlPath := resp.ViewURL
		if urlPath == "" {
			urlPath = resp.PlotURL
		}
		
		if urlPath != "" {
			fullURL := fmt.Sprintf("%s%s", ps.baseURL, urlPath)
			if err := ps.OpenInBrowser(fullURL); err != nil {
				// Don't fail the whole operation if browser opening fails
				fmt.Printf("Warning: Failed to open browser automatically: %v\n", err)
				fmt.Printf("Please open manually: %s\n", fullURL)
			}
		}
	}
	
	return resp, nil
}

// GenerateAndSendAllPlotsWithBrowser generates all plots using batch endpoint and opens dashboard
func (ps *PlottingService) GenerateAndSendAllPlotsWithBrowser(collector *VisualizationCollector) map[PlotType]*PlottingResponse {
	if !ps.enabled {
		return make(map[PlotType]*PlottingResponse)
	}
	
	// Generate all plot data first
	plotTypes := []PlotType{
		TrainingCurves,
		LearningRateSchedule,
		ROCCurve,
		PrecisionRecall,
		ConfusionMatrixPlot,
		RegressionScatter,
		ResidualPlot,
		QQPlot,
		FeatureImportancePlot,
		LearningCurvePlot,
		ValidationCurvePlot,
	}
	
	var plotDataList []PlotData
	results := make(map[PlotType]*PlottingResponse)
	
	// Collect plot data for successful plots
	for _, plotType := range plotTypes {
		var plotData PlotData
		
		switch plotType {
		case TrainingCurves:
			plotData = collector.GenerateTrainingCurvesPlot()
		case LearningRateSchedule:
			plotData = collector.GenerateLearningRateSchedulePlot()
		case ROCCurve:
			plotData = collector.GenerateROCCurvePlot()
		case PrecisionRecall:
			plotData = collector.GeneratePrecisionRecallPlot()
		case ConfusionMatrixPlot:
			plotData = collector.GenerateConfusionMatrixPlot()
		case RegressionScatter:
			plotData = collector.GenerateRegressionScatterPlot()
		case ResidualPlot:
			plotData = collector.GenerateResidualPlot()
		case QQPlot:
			plotData = collector.GenerateQQPlot()
		case FeatureImportancePlot:
			plotData = collector.GenerateFeatureImportancePlot()
		case LearningCurvePlot:
			plotData = collector.GenerateLearningCurvePlot()
		case ValidationCurvePlot:
			plotData = collector.GenerateValidationCurvePlot()
		case PredictionIntervalPlot:
			plotData = collector.GeneratePredictionIntervalPlot()
		case FeatureCorrelationPlot:
			plotData = collector.GenerateFeatureCorrelationPlot()
		case PartialDependencePlot:
			plotData = collector.GeneratePartialDependencePlot()
		default:
			results[plotType] = &PlottingResponse{
				Success: false,
				Message: fmt.Sprintf("Unsupported plot type: %s", plotType),
			}
			continue
		}
		
		// Check if plot data is valid
		if len(plotData.Series) == 0 {
			results[plotType] = &PlottingResponse{
				Success: false,
				Message: fmt.Sprintf("No data available for plot type: %s", plotType),
			}
		} else {
			plotDataList = append(plotDataList, plotData)
			// Don't mark as successful yet - wait for actual batch response
			results[plotType] = &PlottingResponse{
				Success: false,
				Message: "Prepared for batch sending",
			}
		}
	}
	
	// Send as batch if we have any plots
	if len(plotDataList) > 0 {
		batchResp, err := ps.BatchSendPlots(plotDataList)
		if err != nil {
			fmt.Printf("Failed to send batch plots: %v\n", err)
			// Mark all prepared plots as failed
			for plotType, result := range results {
				if result.Message == "Prepared for batch sending" {
					results[plotType] = &PlottingResponse{
						Success: false,
						Message: fmt.Sprintf("Batch send failed: %v", err),
					}
				}
			}
			return results
		}
		
		if batchResp.Success {
			fmt.Printf("✅ Successfully sent %d plots to sidecar (batch ID: %s)\n", batchResp.Summary.Successful, batchResp.BatchID)
			fmt.Printf("🔍 Debug: Dashboard URL from response: '%s'\n", batchResp.DashboardURL)
			
			// Map batch results back to our plot types
			plotTypeIndex := 0
			for plotType, result := range results {
				if result.Message == "Prepared for batch sending" {
					if plotTypeIndex < len(batchResp.Results) {
						batchResult := batchResp.Results[plotTypeIndex]
						results[plotType] = &PlottingResponse{
							Success:      batchResult.Success,
							Message:      "Successfully sent in batch",
							PlotID:       batchResult.PlotID,
							PlotURL:      batchResult.PlotURL,
							ViewURL:      batchResult.ViewURL,
							BatchID:      batchResp.BatchID,
							DashboardURL: batchResp.DashboardURL,
						}
						plotTypeIndex++
					}
				}
			}
			
			// Try to open dashboard if available
			if batchResp.DashboardURL != "" {
				dashboardURL := fmt.Sprintf("%s%s", ps.baseURL, batchResp.DashboardURL)
				fmt.Printf("🔍 Debug: Full dashboard URL: '%s'\n", dashboardURL)
				if err := ps.OpenInBrowser(dashboardURL); err != nil {
					fmt.Printf("Warning: Failed to open dashboard automatically: %v\n", err)
					fmt.Printf("Please open manually: %s\n", dashboardURL)
				} else {
					fmt.Println("🌐 Dashboard opened in browser")
				}
			} else {
				fmt.Printf("📊 No dashboard URL returned - checking individual plots\n")
				
				// If no dashboard, try to open the first successful individual plot
				for _, result := range results {
					if result.Success && result.ViewURL != "" {
						plotURL := fmt.Sprintf("%s%s", ps.baseURL, result.ViewURL)
						fmt.Printf("🔍 Debug: Opening individual plot: %s\n", plotURL)
						if err := ps.OpenInBrowser(plotURL); err != nil {
							fmt.Printf("Warning: Failed to open plot automatically: %v\n", err)
						} else {
							fmt.Println("🌐 First plot opened in browser")
							break
						}
					}
				}
			}
		} else {
			fmt.Printf("Failed to generate batch plots: %s\n", batchResp.Message)
			// Mark all prepared plots as failed
			for plotType, result := range results {
				if result.Message == "Prepared for batch sending" {
					results[plotType] = &PlottingResponse{
						Success: false,
						Message: fmt.Sprintf("Batch generation failed: %s", batchResp.Message),
					}
				}
			}
		}
	} else {
		fmt.Println("No valid plots to display")
	}
	
	return results
}