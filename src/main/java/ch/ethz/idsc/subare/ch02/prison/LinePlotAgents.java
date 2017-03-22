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
  @Override
  public void start(Stage stage) {
    ListPlot listPlot = new ListPlot();
    {
      Agent a1;
      Agent a2;
      a1 = new UCBAgent(2, RealScalar.of(1.3));
      a2 = new UCBAgent(2, RealScalar.of(1.1));
//      a1 = new OptimistAgent(2, RealScalar.of(6), RealScalar.of(.1));
//      a2 = new OptimistAgent(2, RealScalar.of(5), RealScalar.of(.3));
      Training.train(a1, a2, 100);
      System.out.println(a1 + " " + a1.getAverage().number().doubleValue());
      System.out.println(a2 + " " + a2.getAverage().number().doubleValue());
      if (true) {
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