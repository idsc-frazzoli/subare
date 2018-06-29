// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.ch04.gambler.Gambler;
import ch.ethz.idsc.subare.ch04.grid.Gridworld;
import ch.ethz.idsc.subare.ch05.infvar.InfiniteVariance;
import ch.ethz.idsc.subare.ch05.racetrack.RacetrackHelper;
import ch.ethz.idsc.subare.ch05.wireloop.WireloopHelper;
import ch.ethz.idsc.subare.ch05.wireloop.WireloopReward;
import ch.ethz.idsc.subare.ch06.cliff.Cliffwalk;
import ch.ethz.idsc.subare.ch06.maxbias.Maxbias;
import ch.ethz.idsc.subare.ch06.windy.Windygrid;
import ch.ethz.idsc.subare.ch08.maze.DynamazeHelper;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.demo.airport.Airport;

public enum AnalysisModels {
  AIRPORT() {
    @Override
    public MonteCarloInterface supply() {
      return new Airport();
    }
  },
  GAMBLER() {
    @Override
    public MonteCarloInterface supply() {
      return Gambler.createDefault();
    }
  }, //
  MAZE() {
    @Override
    public MonteCarloInterface supply() {
      return DynamazeHelper.create5(3);
    }
  }, //
  WIRELOOP() {
    @Override
    public MonteCarloInterface supply() throws Exception {
      WireloopReward wireloopReward = WireloopReward.freeSteps();
      wireloopReward = WireloopReward.constantCost();
      return WireloopHelper.create("wirec", WireloopReward::id_x, wireloopReward);
    }
  }, //
  GRIDWORLD() {
    @Override
    public MonteCarloInterface supply() {
      return new Gridworld();
    }
  },
  INFINITEVARIANCE() {
    @Override
    public MonteCarloInterface supply() throws Exception {
      return new InfiniteVariance();
    }
  }, //
  RACETRACK() {
    @Override
    public MonteCarloInterface supply() throws Exception {
      return RacetrackHelper.create("track0", 5);
    }
  }, //
  CLIFFWALK() {
    @Override
    public MonteCarloInterface supply() throws Exception {
      return new Cliffwalk(12, 4);
    }
  }, //
  MAXBIAS() {
    @Override
    public MonteCarloInterface supply() throws Exception {
      return new Maxbias(5);
    }
  }, //
  WINDYGRID() {
    @Override
    public MonteCarloInterface supply() throws Exception {
      return Windygrid.createFour();
    }
  },//
  ;
  // ---
  public abstract MonteCarloInterface supply() throws Exception;
}
