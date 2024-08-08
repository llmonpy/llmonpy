
export let API = null;

export function InitLLMonPyScopeAPI(apiUrl) {
  API = new LLMonPyScopeAPI(apiUrl);
}

export class LLMonPyScopeAPI {
  constructor(apiUrl) {
    this.apiUrl = apiUrl;
    console.log("apiUrl:" + apiUrl);
    this.testApPI();
  }

  async testApPI() {
    const response = await fetch(this.apiUrl + '/hello_world');
    const data = await response.json();
    console.log(data);
  }

  async getTraceList() {
    const response = await fetch(this.apiUrl + '/get_trace_list');
    const data = await response.json();
    return data;
  }

  async getCompleteTrace(traceId) {
    const url = this.apiUrl + '/get_complete_trace_by_id?trace_id=' + traceId;
    const response = await fetch(url);
    const data = await response.json();
    return data;
  }

  async getTourneyStepNameList() {
    const response = await fetch(this.apiUrl + '/get_tourney_step_name_list');
    const data = await response.json();
    return data;
  }

  async getTourneyResultsForStepName(stepName) {
    const url = this.apiUrl + '/get_tourney_results_for_step_name?step_name=' + stepName;
    const response = await fetch(url);
    const data = await response.json();
    return data;
  }

  async getEventsForStep(stepId) {
    const url = this.apiUrl + '/get_events_for_step?step_id=' + stepId;
    const response = await fetch(url);
    const data = await response.json();
    return data;
  }
}

export class DisplayTrace {
  constructor(trace) {
    this.trace = trace;
    this.traceId = trace.trace_id;
    this.traceName = DisplayTrace.generateDisplayName(trace);
  }

  static generateDisplayName(trace) {
    const lastPeriod = trace.title.lastIndexOf('.');
    let result = trace.title.substring(lastPeriod + 1) + " " + trace.start_time;
    return result;
  }

  static CreateDisplayTraceList(traceList) {
    let result = [];
    for (let trace of traceList) {
      result.push(new DisplayTrace(trace));
    }
    return result;
  }
}

export class DisplayStep {
  constructor(step, parentStep, tourneyResults, componentName) {
    this.step = step;
    this.displayName = this.generateDisplayName(step.step_name);
    this.parentStep = parentStep;
    this.tourneyResults = tourneyResults;
    this.componentName = componentName;
    this.children = [];
  }

  generateDisplayName(fullName) {
    const lastPeriod = fullName.lastIndexOf('.');
    let result = fullName.substring(lastPeriod + 1);
    return result;
  }

  addChild(child) {
    this.children.push(child);
  }

  static CreateDisplayStepList(stepList, tourneyResultList, componentMap, defaultComponentName) {
    let result = [];
    let stepMap = new Map();
    // stepList is ordered by step_index, which means we will see parent steps before child steps
    for (let step of stepList) {
      let parentStep = stepMap.get(step.parent_step_id);
      let componentName = defaultComponentName;
      if (step.step_type in componentMap) {
        componentName = componentMap[step.step_type];
      }
      const tourneyResults = DisplayStep.FindTourneyResultsForStepId(step.step_id, tourneyResultList);
      const displayStep = new DisplayStep(step, parentStep, tourneyResults, componentName);
      result.push(displayStep);
      stepMap.set(step.step_id, displayStep);
      if (parentStep != null) {
        parentStep.addChild(displayStep);
      }
    }
    return result;
  }

  static FindTourneyResultsForStepId(stepId, tourneyResultList) {
    let result = null;
    for (let tourneyResult of tourneyResultList) {
      if (tourneyResult.step_id == stepId) {
        result = tourneyResult;
        break;
      }
    }
    return null;
  }

  static FlattenStepList(flattendStepList, displayStepList) {
    if (displayStepList != null ) {
      for (let displayStep of displayStepList) {
        flattendStepList.push(displayStep.step);
        DisplayStep.FlattenStepList(flattendStepList, displayStep.children);
      }
    }
  }
}

export function LLMClientSettingsToString(settings) {
  let result = "";
  if (settings != null) {
    let addSeparator = false;
    result += "{ "
    for (let key in settings) {
      if (addSeparator) {
        result += ", ";
      }
      result += key + ":" + settings[key];
    }
    result += " }";
  }
  return result;
}

export function CalculateDuration(start_iso_8601_string, end_iso_8601_string) {
  let startDate = new Date(start_iso_8601_string);
  let endDate = new Date(end_iso_8601_string);
  let duration = (endDate - startDate) / 1000; // seconds
  console.log("duration:" + duration);
  return duration;
}

export class ModelReport {
  constructor(modelInfo) {
    this.fullName = ModelReport.GenerateFullName(modelInfo);
    this.cost = 0;
    this.victoryCount = 0;
    this.costPerVictory = Number.MAX_VALUE;
  }

  addVictoryCount(count, cost) {
    this.victoryCount += count;
    this.cost += cost;
  }

  finish() {
    if (this.victoryCount > 0) {
      this.costPerVictory = this.cost / this.victoryCount;
    }
  }

  getCostPerVictoryString() {
    let result = ""
    if (this.victoryCount != 0) {
      result = new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 5
      }).format(this.costPerVictory);
    } else {
      result = new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 5
      }).format(this.cost) + " (no victories)";
    }
    return result;
  }

  static GenerateFullName(modelInfo) {
    const modelName = modelInfo.model_name;
    const settingsString = LLMClientSettingsToString(modelInfo.client_settings_dict);
    const result = modelName + " " + settingsString;
    return result;
  }

  static GenerateModelReportList(displayStepList) {
    let result = null;
    if ( displayStepList != null && displayStepList.length > 0) {
      const flattendStepList = [];
      DisplayStep.FlattenStepList(flattendStepList, displayStepList);
      let modelReportMap = new Map();
      let stepIdMap = new Map();
      for (let step of flattendStepList) {
        stepIdMap.set(step.step_id, step);
      }
      for (let step of flattendStepList) {
        if (step.step_type == "ranker") {
          const outputList = step.output_dict.ordered_response_list;
          for (const judgedOutput of outputList) {
            const judgedStep = stepIdMap.get(judgedOutput.step_id);
            if ( judgedStep != null) { // would be null if the parent step was a ranker step, generation happened elsewhere
              const modelInfo = judgedOutput.llm_model_info;
              let report = modelReportMap.get(ModelReport.GenerateFullName(modelInfo));
              if (report == null) {
                report = new ModelReport(modelInfo);
                modelReportMap.set(report.fullName, report);
              }
              report.addVictoryCount(judgedOutput.victory_count, judgedStep.cost);
            }
          }
        }
      }
      result = [];
      for (let modelReport of modelReportMap.values()) {
        if (modelReport.cost > 0) {
          modelReport.finish();
          result.push(modelReport);
        }
      }
      result.sort((a, b) => {
        if (a.costPerVictory < b.costPerVictory) {
          return -1;
        }
        if (a.costPerVictory > b.costPerVictory) {
          return 1;
        }
        return 0;
      });
      result = result.length == 0 ? null : result;
    }
    return result;
  }
}

