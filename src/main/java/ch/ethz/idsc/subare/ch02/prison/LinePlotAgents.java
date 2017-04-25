// code by jph
package ch.ethz.idsc.subare.ch02.prison;

import java.util.Arrays;
import java.util.List;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.ch02.GradientAgent;
import ch.ethz.idsc.subare.ch02.OptimistAgent;
import ch.ethz.idsc.subare.ch02.UCBAgent;
import ch.ethz.idsc.tensor.RationalScalar;
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
  Agent a1;
  Agent a2;

  void initOptimists() {
    a1 = new OptimistAgent(ACTION_COUNT, RationalScalar.of(40, 10), RationalScalar.of(10, 100));
    a1.setOpeningSequence(0);
    a2 = new OptimistAgent(ACTION_COUNT, RationalScalar.of(38, 10), RationalScalar.of(10, 100));
    a2.setOpeningSequence(1);
  }

  void initOptiTFT() {
    a1 = new OptimistAgent(ACTION_COUNT, RationalScalar.of(60, 10), RationalScalar.of(10, 100));
    a1.setOpeningSequence(1);
    a2 = new TitForTatAgent();
  }

  void initUCB() {
    a1 = new UCBAgent(ACTION_COUNT, RationalScalar.of(10, 10));
    a1.setOpeningSequence(1);
    // a2 = new UCBAgent(ACTION_COUNT, RationalScalar.of(9716, 10000)); // <- approx winwin threshold
    a2 = new UCBAgent(ACTION_COUNT, RationalScalar.of(10000, 10000)); // <- approx winwin threshold
    a2.setOpeningSequence(0);
  }

  void initUCBTFT() {
    a1 = new UCBAgent(ACTION_COUNT, RationalScalar.of(10, 10));
    a1.setOpeningSequence(1);
    a2 = new TitForTatAgent();
  }

  void initGradient() {
    a1 = new GradientAgent(ACTION_COUNT, RealScalar.of(.1));
    a2 = new GradientAgent(ACTION_COUNT, RealScalar.of(.1));
  }

  @Override
  public void start(Stage stage) {
    ListPlot listPlot = new ListPlot();
    // initUCB();
    // initOptimists();
    // initOptiTFT();
    initUCBTFT();
    // initGradient();
    // ---
    Training.train(a1, a2, 100);
    // ---
    System.out.println(SummaryString.of(a1));
    System.out.println(SummaryString.of(a2));
    // a1.getQValues().flatten(0).forEach(System.out::println);
    // ---
    List<Agent> list = Arrays.asList(a1, a2);
    for (int index = 0; index < list.size(); ++index) {
      Agent agent = list.get(index);
      XYChart.Series<Number, Number> series = listPlot.addVector(agent.getActions());
      series.setName("action A" + index);
    }
    for (int index = 0; index < list.size(); ++index) {
      Agent agent = list.get(index);
      Tensor qt = Transpose.of(agent.getQValues());
      for (int a = 0; a < ACTION_COUNT; ++a) {
        XYChart.Series<Number, Number> series = listPlot.addVector(qt.get(a));
        series.setName("A" + index + " Q(" + a + ")");
      }
    }
    stage.setTitle("Two Agents");
    // listPlot.xAxis.setLabel("time");
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