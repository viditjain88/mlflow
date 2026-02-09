from hmdss.tools.ann_tool import ANNPredictionTool

def test_ann_tool():
    tool = ANNPredictionTool()
    print(f"Tool Name: {tool.name}")
    prediction = tool._run("2023-12-25")
    print(prediction)
    assert "Predicted Patient Volume" in prediction

if __name__ == "__main__":
    test_ann_tool()
