#include <iostream>
#include <string>
#include <cstdlib>

class MetricRunner {
public:
    void runEvaluation(const std::string& dataset_path) {
        std::cout << "Checking " << dataset_path << std::endl;
        
        std::string command = "python3 infering.py trained_model.pkl " + dataset_path;
        std::string result = executeCommand(command);
        
        parseAndPrintMetrics(result);
    }
    
private:
    std::string executeCommand(const std::string& command) {
        std::string result = "";
        char buffer[128];
        FILE* pipe = popen(command.c_str(), "r");
        
        if (!pipe) return "{\"error\": \"Command failed\"}";
        
        while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
            result += buffer;
        }
        
        pclose(pipe);
        return result;
    }
    
    void parseAndPrintMetrics(const std::string& jsonResult) {
        if (jsonResult.find("\"error\"") != std::string::npos) {
            size_t error_start = jsonResult.find("\"error\":") + 8;
            size_t error_end = jsonResult.find("\"", error_start);
            std::string error_msg = jsonResult.substr(error_start, error_end - error_start);
            
            std::cout << "\nERROR:" << std::endl;
            std::cout << "  " << error_msg << std::endl;
            return;
        }
        
        auto getValue = [&](const std::string& key) -> std::string {
            size_t pos = jsonResult.find("\"" + key + "\":");
            if (pos == std::string::npos) return "N/A";
            size_t start = jsonResult.find(":", pos) + 1;
            size_t end = jsonResult.find(",", start);
            if (end == std::string::npos) end = jsonResult.find("}", start);
            std::string value = jsonResult.substr(start, end - start);
            
            value.erase(0, value.find_first_not_of(" \t\n\r\""));
            value.erase(value.find_last_not_of(" \t\n\r\"") + 1);
            return value;
        };
        
        std::string accuracy = getValue("accuracy");
        std::string precision = getValue("precision");
        std::string recall = getValue("recall");
        std::string f1 = getValue("f1_score");
        std::string total = getValue("total_records");
        std::string actual = getValue("actual_attacks");
        std::string predicted = getValue("predicted_attacks");
        std::string ddos_count = getValue("ddos_count");
        std::string portscan_count = getValue("portscan_count");
        std::string normal_count = getValue("normal_count");
        
        std::cout << "Metrics:" << std::endl;
        std::cout << "  Accuracy:  " << accuracy << std::endl;
        std::cout << "  Precision: " << precision << std::endl;
        std::cout << "  Recall:    " << recall << std::endl;
        std::cout << "  F1-Score:  " << f1 << std::endl;
        
        std::cout << "\nStatistics:" << std::endl;
        std::cout << "  Total records: " << total << std::endl;
        std::cout << "  Actual attacks: " << actual << std::endl;
        std::cout << "  Predicted attacks: " << predicted << std::endl;
        
        if (ddos_count != "N/A" || portscan_count != "N/A") {
            std::cout << "\nAttack Types:" << std::endl;
            if (ddos_count != "N/A") {
                std::cout << "  DDoS attacks:    " << ddos_count << std::endl;
            }
            if (portscan_count != "N/A") {
                std::cout << "  PortScan attacks: " << portscan_count << std::endl;
            }
            if (normal_count != "N/A") {
                std::cout << "  Normal traffic:   " << normal_count << std::endl;
            }
        }
        
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: ./detect dataset.csv" << std::endl;
        return 1;
    }
    
    MetricRunner detector;
    detector.runEvaluation(argv[1]);
    return 0;
}
