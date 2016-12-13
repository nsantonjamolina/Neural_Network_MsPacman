package pacman.entries.pacman;

import NeuralNetwork.NeuralNetwork;
import dataRecording.DataTuple;
import pacman.controllers.Controller;
import pacman.game.Constants;
import pacman.game.Constants.GHOST;
import pacman.game.Constants.MOVE;
import pacman.game.Constants.STRATEGY;
import pacman.game.Game;

import java.util.ArrayList;


/*
 * This is the class you need to modify for your entry. In particular, you need to
 * fill in the getAction() method. Any additional classes you write should either
 * be placed in this package or sub-packages (e.g., game.entries.pacman.mypackage).
 */
public class MyPacMan extends Controller<MOVE>
{
    public NeuralNetwork neuralNetwork;

	public MyPacMan() {
		super();

        int input = 2;
        int hide = 2 * input + 1;
        int output = 1;

        neuralNetwork = new NeuralNetwork(input, hide, output, 70);
        neuralNetwork.Train();

	}
	private MOVE myMove=MOVE.NEUTRAL;
	
	public MOVE getMove(Game game, long timeDue) 
	{
		//Place your game logic here to play the game as Ms Pac-Man
        DataTuple currentGameTuple = new DataTuple(game, myMove);


        int minDistGRunAway = Integer.MAX_VALUE;
        int ghostEdibleTimeRemaining = 0;
        GHOST ghostDelQueHuir = null;


        for (GHOST ghost : GHOST.values()) {
            int distance = game.getShortestPathDistance(currentGameTuple.pacmanPosition, game.getGhostCurrentNodeIndex(ghost));
            if(distance < minDistGRunAway)  {
                minDistGRunAway = distance;
                ghostDelQueHuir = ghost;
                ghostEdibleTimeRemaining = game.getGhostEdibleTime(ghost);
            }
        }


        float[] inputs = {(float) currentGameTuple.normalizeDistance(minDistGRunAway), (float) currentGameTuple.normalizeEdibleTime(ghostEdibleTimeRemaining)};
        float output = neuralNetwork.forwardPropagation(inputs);
        STRATEGY outputStrategy = null;

        if(output >= .01f && output <= .25f) {
            outputStrategy = STRATEGY.CHASE;
        } else if(output > .25f && output <= .75f) {
            outputStrategy = STRATEGY.EATPILLS;
        } else if(output > .75f && output <= .99f) {
            outputStrategy = STRATEGY.RUNAWAY;
        } else {
            outputStrategy = STRATEGY.NEUTRAL;
        }


        switch (outputStrategy) {

            case EATPILLS:
                int[] pills=game.getPillIndices();
                int[] powerPills=game.getPowerPillIndices();

                ArrayList<Integer> targets=new ArrayList<Integer>();

                for(int i=0;i<pills.length;i++)
                    if(game.isPillStillAvailable(i))
                        targets.add(pills[i]);

                for(int i=0;i<powerPills.length;i++)
                    if(game.isPowerPillStillAvailable(i))
                        targets.add(powerPills[i]);

                int[] targetsArray=new int[targets.size()];

                for(int i=0;i<targetsArray.length;i++)
                    targetsArray[i]=targets.get(i);

                myMove = game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(),game.getClosestNodeIndexFromNodeIndex(game.getPacmanCurrentNodeIndex(),targetsArray, Constants.DM.PATH), Constants.DM.PATH);
                /*if (killerPacManResult == nextMove) numberOfCorrectDecisions++;
                else System.err.println("Didnt make the same decision as KillerPacMan, KillerPacMan chose: " + killerPacManResult.toString());*/
                return myMove;
            case CHASE:
                int minDistGChase = Integer.MAX_VALUE;
                GHOST ghostAPerseguir = null;

                for (GHOST ghost : GHOST.values()) {
                    int distance = game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(ghost));
                    if(distance < minDistGChase)  {
                        minDistGChase = distance;
                        ghostAPerseguir = ghost;
                    }
                }

                myMove = game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(),game.getGhostCurrentNodeIndex(ghostAPerseguir), Constants.DM.PATH);
                /*if (killerPacManResult == nextMove) numberOfCorrectDecisions++;*/
                return myMove;
            case RUNAWAY:
                int minDistGRunAwayy = Integer.MAX_VALUE;
                GHOST ghostDelQueHuirr = null;

                for (GHOST ghost : GHOST.values()) {
                    int distance = game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(ghost));
                    if(distance < minDistGRunAway)  {
                        minDistGRunAwayy = distance;
                        ghostDelQueHuirr = ghost;
                    }
                }

                myMove = game.getNextMoveAwayFromTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(ghostDelQueHuir), Constants.DM.PATH);
                /*if (killerPacManResult == nextMove) numberOfCorrectDecisions++;*/
                return myMove;
            default:
                return MOVE.NEUTRAL;
        }
	}
}