package ch.ethz.idsc.subare.ch02.prison;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.ch02.OptimistAgent;
import ch.ethz.idsc.subare.ch02.UCBAgent;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Transpose;
import charts.ListPlot;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

@SuppressWarnings("restriction")
public class LinePlotAgents extends Application {
  public static final int ACTION_COUNT = 2;

  public String summaryString(Agent agent) {
    return String.format("%25s  %6.3f  %5d RND", //
        agent.toString(), agent.getRewardAverage().number(), agent.getRandomizedDecisionCount());
  }

  Agent a1;
  Agent a2;

  void initOptimists() {
    a1 = new OptimistAgent(ACTION_COUNT, RealScalar.of(3.8), RealScalar.of(.1));
    a1.setOpeningSequence(0);
    a2 = new OptimistAgent(ACTION_COUNT, RealScalar.of(4), RealScalar.of(.1));
    a2.setOpeningSequence(1);
  }

  void initUCB() {
    a1 = new UCBAgent(ACTION_COUNT, RealScalar.of(3.8));
    a1.setOpeningSequence(0);
    a2 = new UCBAgent(ACTION_COUNT, RealScalar.of(4));
    a2.setOpeningSequence(0);
  }

  @Override
  public void start(Stage stage) {
    ListPlot listPlot = new ListPlot();
    {
      // 1.2, 2
      // a1 = new GradientAgent(2, RealScalar.of(.1));
      // a2 = new GradientAgent(2, RealScalar.of(.1));
      // initUCB();
      initOptimists();
      // ---
      Training.train(a1, a2, 200);
      // ---
      System.out.println(summaryString(a1));
      System.out.println(summaryString(a2));
      // ---
      {
        {
          XYChart.Series<Number, Number> s1 = listPlot.addVector(a1.getActions());
          s1.setName("action A1");
        }
        {
          XYChart.Series<Number, Number> s2 = listPlot.addVector(a2.getActions());
          s2.setName("action A2");
        }
      }
      {
        // System.out.println(Pretty.of(a1.getQValues()));
        Tensor qt = Transpose.of(a1.getQValues());
        {
          XYChart.Series<Number, Number> s2 = listPlot.addVector(qt.get(0));
          s2.setName("A1 Q(0)");
        }
        {
          XYChart.Series<Number, Number> s2 = listPlot.addVector(qt.get(1));
          s2.setName("A1 Q(1)");
        }
      }
      {
        // System.out.println(Pretty.of(a1.getQValues()));
        Tensor qt = Transpose.of(a2.getQValues());
        {
          XYChart.Series<Number, Number> s2 = listPlot.addVector(qt.get(0));
          s2.setName("A2 Q(0)");
        }
        {
          XYChart.Series<Number, Number> s2 = listPlot.addVector(qt.get(1));
          s2.setName("A2 Q(1)");
        }
      }
    }
    stage.setTitle("Two Agents");
    listPlot.xAxis.setLabel("Number of action");
    {
      Scene scene = new Scene(listPlot.lineChart, 1600, 400);
      stage.setScene(scene);
      stage.show();
    }
  }

  public static void main(String[] args) {
    launch(args);
  }
}