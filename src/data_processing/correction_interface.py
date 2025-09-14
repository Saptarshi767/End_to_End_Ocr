"""
Manual correction interface for table data validation and editing.
"""

from typing import List, Dict, Any, Optional, Callable
import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

from ..core.models import Table, ValidationResult
from .table_validator import TableValidator, TableValidationReport, CorrectionAction, CellValidation

logger = logging.getLogger(__name__)


@dataclass
class CorrectionSession:
    """Represents a table correction session."""
    session_id: str
    table_id: str
    original_table: Table
    current_table: Table
    validation_report: TableValidationReport
    applied_corrections: List[CorrectionAction]
    created_at: datetime
    last_modified: datetime
    user_id: Optional[str] = None


@dataclass
class CorrectionSuggestion:
    """Represents a correction suggestion for the UI."""
    row: int
    column: int
    original_value: str
    suggested_value: str
    confidence: float
    reason: str
    accepted: bool = False


class TableCorrectionInterface:
    """
    Interface for manual table data correction and validation.
    """
    
    def __init__(self, validator: Optional[TableValidator] = None):
        self.validator = validator or TableValidator()
        self.active_sessions: Dict[str, CorrectionSession] = {}
        self.correction_callbacks: List[Callable] = []
    
    def start_correction_session(self, table: Table, user_id: Optional[str] = None) -> str:
        """
        Start a new correction session for a table.
        
        Args:
            table: Table to correct
            user_id: Optional user identifier
            
        Returns:
            Session ID for the correction session
        """
        try:
            logger.info(f"Starting correction session for table {table.id}")
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Validate the table
            validation_report = self.validator.validate_table_structure(table)
            
            # Create correction session
            session = CorrectionSession(
                session_id=session_id,
                table_id=table.id,
                original_table=table,
                current_table=Table(
                    id=table.id,
                    headers=table.headers.copy(),
                    rows=[row.copy() for row in table.rows],
                    confidence=table.confidence,
                    region=table.region,
                    metadata=table.metadata.copy()
                ),
                validation_report=validation_report,
                applied_corrections=[],
                created_at=datetime.now(),
                last_modified=datetime.now(),
                user_id=user_id
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Correction session {session_id} started successfully")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start correction session: {str(e)}")
            raise
    
    def get_correction_suggestions(self, session_id: str) -> List[CorrectionSuggestion]:
        """
        Get correction suggestions for a session.
        
        Args:
            session_id: Correction session ID
            
        Returns:
            List of correction suggestions
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        suggestions = []
        
        for cell_validation in session.validation_report.cell_validations:
            if cell_validation.suggested_value:
                suggestion = CorrectionSuggestion(
                    row=cell_validation.row,
                    column=cell_validation.column,
                    original_value=cell_validation.original_value,
                    suggested_value=cell_validation.suggested_value,
                    confidence=cell_validation.confidence,
                    reason='; '.join(cell_validation.issues)
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def apply_cell_correction(self, session_id: str, row: int, column: int, 
                            new_value: str, reason: str = "Manual correction") -> bool:
        """
        Apply a correction to a specific cell.
        
        Args:
            session_id: Correction session ID
            row: Row index
            column: Column index
            new_value: New cell value
            reason: Reason for the correction
            
        Returns:
            True if correction was applied successfully
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            # Validate indices
            if not (0 <= row < len(session.current_table.rows)):
                raise ValueError(f"Invalid row index: {row}")
            
            if not (0 <= column < len(session.current_table.rows[row])):
                raise ValueError(f"Invalid column index: {column}")
            
            # Get original value
            original_value = session.current_table.rows[row][column]
            
            # Apply correction
            session.current_table.rows[row][column] = new_value
            
            # Record the correction
            correction = CorrectionAction(
                action_type="replace",
                row=row,
                column=column,
                original_value=original_value,
                new_value=new_value,
                confidence=1.0,  # Manual corrections have high confidence
                reason=reason
            )
            
            session.applied_corrections.append(correction)
            session.last_modified = datetime.now()
            
            # Notify callbacks
            self._notify_correction_callbacks(session_id, correction)
            
            logger.info(f"Applied correction to cell ({row}, {column}) in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply cell correction: {str(e)}")
            return False
    
    def apply_bulk_corrections(self, session_id: str, 
                             corrections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply multiple corrections at once.
        
        Args:
            session_id: Correction session ID
            corrections: List of correction dictionaries
            
        Returns:
            Dictionary with application results
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            results = {
                "applied": 0,
                "failed": 0,
                "errors": []
            }
            
            for correction_data in corrections:
                try:
                    success = self.apply_cell_correction(
                        session_id=session_id,
                        row=correction_data["row"],
                        column=correction_data["column"],
                        new_value=correction_data["new_value"],
                        reason=correction_data.get("reason", "Bulk correction")
                    )
                    
                    if success:
                        results["applied"] += 1
                    else:
                        results["failed"] += 1
                        
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Row {correction_data.get('row', '?')}, "
                                          f"Column {correction_data.get('column', '?')}: {str(e)}")
            
            logger.info(f"Bulk corrections applied: {results['applied']} successful, "
                       f"{results['failed']} failed")
            return results
            
        except Exception as e:
            logger.error(f"Failed to apply bulk corrections: {str(e)}")
            return {"applied": 0, "failed": len(corrections), "errors": [str(e)]}
    
    def accept_suggestion(self, session_id: str, row: int, column: int) -> bool:
        """
        Accept a correction suggestion.
        
        Args:
            session_id: Correction session ID
            row: Row index
            column: Column index
            
        Returns:
            True if suggestion was accepted
        """
        try:
            # Find the suggestion
            suggestions = self.get_correction_suggestions(session_id)
            
            for suggestion in suggestions:
                if suggestion.row == row and suggestion.column == column:
                    return self.apply_cell_correction(
                        session_id=session_id,
                        row=row,
                        column=column,
                        new_value=suggestion.suggested_value,
                        reason=f"Accepted suggestion: {suggestion.reason}"
                    )
            
            logger.warning(f"No suggestion found for cell ({row}, {column}) in session {session_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to accept suggestion: {str(e)}")
            return False
    
    def reject_suggestion(self, session_id: str, row: int, column: int) -> bool:
        """
        Reject a correction suggestion.
        
        Args:
            session_id: Correction session ID
            row: Row index
            column: Column index
            
        Returns:
            True if suggestion was rejected
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            # Find and mark the cell validation as rejected
            for cell_validation in session.validation_report.cell_validations:
                if cell_validation.row == row and cell_validation.column == column:
                    # Add rejection marker to issues
                    if "User rejected suggestion" not in cell_validation.issues:
                        cell_validation.issues.append("User rejected suggestion")
                    break
            
            session.last_modified = datetime.now()
            
            logger.info(f"Rejected suggestion for cell ({row}, {column}) in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reject suggestion: {str(e)}")
            return False
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current status of a correction session.
        
        Args:
            session_id: Correction session ID
            
        Returns:
            Dictionary with session status information
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Count suggestions
        suggestions = self.get_correction_suggestions(session_id)
        pending_suggestions = len([s for s in suggestions if not s.accepted])
        
        return {
            "session_id": session_id,
            "table_id": session.table_id,
            "created_at": session.created_at.isoformat(),
            "last_modified": session.last_modified.isoformat(),
            "corrections_applied": len(session.applied_corrections),
            "pending_suggestions": pending_suggestions,
            "total_suggestions": len(suggestions),
            "validation_score": session.validation_report.overall_confidence,
            "is_valid": session.validation_report.is_valid,
            "structure_issues": len(session.validation_report.structure_issues),
            "data_quality_issues": len(session.validation_report.data_quality_issues)
        }
    
    def finalize_corrections(self, session_id: str) -> Table:
        """
        Finalize corrections and return the corrected table.
        
        Args:
            session_id: Correction session ID
            
        Returns:
            Corrected table
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            # Create final table with metadata
            final_table = Table(
                id=session.table_id,
                headers=session.current_table.headers.copy(),
                rows=[row.copy() for row in session.current_table.rows],
                confidence=session.current_table.confidence,
                region=session.current_table.region,
                metadata=session.current_table.metadata.copy()
            )
            
            # Update metadata with correction information
            final_table.metadata.update({
                "correction_session_id": session_id,
                "corrections_applied": len(session.applied_corrections),
                "correction_timestamp": datetime.now().isoformat(),
                "original_confidence": session.original_table.confidence,
                "corrected_confidence": final_table.confidence
            })
            
            # Recalculate confidence if corrections were applied
            if session.applied_corrections:
                # Boost confidence based on manual corrections
                confidence_boost = min(0.3, len(session.applied_corrections) * 0.05)
                final_table.confidence = min(1.0, final_table.confidence + confidence_boost)
                final_table.metadata["corrected_confidence"] = final_table.confidence
            
            logger.info(f"Finalized corrections for session {session_id}. "
                       f"Applied {len(session.applied_corrections)} corrections")
            
            return final_table
            
        except Exception as e:
            logger.error(f"Failed to finalize corrections: {str(e)}")
            raise
    
    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Export session data for external use or storage.
        
        Args:
            session_id: Correction session ID
            
        Returns:
            Dictionary with complete session data
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        return {
            "session_info": {
                "session_id": session.session_id,
                "table_id": session.table_id,
                "created_at": session.created_at.isoformat(),
                "last_modified": session.last_modified.isoformat(),
                "user_id": session.user_id
            },
            "original_table": {
                "headers": session.original_table.headers,
                "rows": session.original_table.rows,
                "confidence": session.original_table.confidence,
                "metadata": session.original_table.metadata
            },
            "corrected_table": {
                "headers": session.current_table.headers,
                "rows": session.current_table.rows,
                "confidence": session.current_table.confidence,
                "metadata": session.current_table.metadata
            },
            "validation_report": {
                "overall_confidence": session.validation_report.overall_confidence,
                "is_valid": session.validation_report.is_valid,
                "structure_issues": session.validation_report.structure_issues,
                "data_quality_issues": session.validation_report.data_quality_issues,
                "cell_validations_count": len(session.validation_report.cell_validations)
            },
            "corrections": [asdict(correction) for correction in session.applied_corrections],
            "suggestions": [asdict(suggestion) for suggestion in self.get_correction_suggestions(session_id)]
        }
    
    def close_session(self, session_id: str) -> bool:
        """
        Close and clean up a correction session.
        
        Args:
            session_id: Correction session ID
            
        Returns:
            True if session was closed successfully
        """
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"Closed correction session {session_id}")
                return True
            else:
                logger.warning(f"Session {session_id} not found for closing")
                return False
                
        except Exception as e:
            logger.error(f"Failed to close session {session_id}: {str(e)}")
            return False
    
    def add_correction_callback(self, callback: Callable) -> None:
        """
        Add a callback function to be called when corrections are applied.
        
        Args:
            callback: Function to call with (session_id, correction) parameters
        """
        self.correction_callbacks.append(callback)
    
    def _notify_correction_callbacks(self, session_id: str, correction: CorrectionAction) -> None:
        """Notify all registered callbacks about a correction."""
        for callback in self.correction_callbacks:
            try:
                callback(session_id, correction)
            except Exception as e:
                logger.warning(f"Correction callback failed: {str(e)}")
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of active session IDs
        """
        return list(self.active_sessions.keys())
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old correction sessions.
        
        Args:
            max_age_hours: Maximum age of sessions to keep
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            current_time = datetime.now()
            sessions_to_remove = []
            
            for session_id, session in self.active_sessions.items():
                age_hours = (current_time - session.last_modified).total_seconds() / 3600
                if age_hours > max_age_hours:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
            
            if sessions_to_remove:
                logger.info(f"Cleaned up {len(sessions_to_remove)} old correction sessions")
            
            return len(sessions_to_remove)
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {str(e)}")
            return 0